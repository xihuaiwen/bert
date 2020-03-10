import os
import re
import sys
import json
import time
import cpuinfo
import platform
import subprocess
from functools import reduce

from coolname import generate_slug

from .utils import copy, finish, warn, fail, detect_framework
from .archive import extract_max_memory
from .tf_log import extract_tf_tile_mapping, largest_tile_mapping_file, convert_graph_to_svg, find_xla_dot_file

art = r"""
__________________   ________ ________ _______ ________________________ __________
__  ____/__  ____/   ___  __ \___  __ \__  __ \___  ____/____  _/___  / ___  ____/
_  / __  _  /        __  /_/ /__  /_/ /_  / / /__  /_     __  /  __  /  __  __/
/ /_/ /  / /___      _  ____/ _  _  _/ / /_/ / _  __/    __/ /   _  /____  /___
\____/   \____/      /_/      /_/ |_|  \____/  /_/       /___/   /_____//_____/
"""

def gather_system_info():
    cpu_info = cpuinfo.get_cpu_info()
    # Too verbose
    del cpu_info['flags']
    return dict(
        machine=platform.machine(),
        version=platform.version(),
        platform=platform.platform(),
        system=platform.system(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        linux_dist=platform.dist(),
        mac_ver=platform.mac_ver(),
        cpu_info=cpu_info,
    )

def main(
    program,
    profile_dir='profile',
    collect_sys_info=True,
    verbose=False,
    instrumentation=False,
    pretty_json=False,
    pipe_to_shell=False,
    name=None,
    use_cbor=False, # TODO: Implement this!
):
    print(art)

    # Create profile directory
    profile_dir = os.path.abspath(profile_dir)
    os.makedirs(profile_dir, exist_ok=True)

    if name is None:
        name = generate_slug()
        if verbose:
            print(f"Generated name: {name}")

    # Log files
    archive_file = os.path.join(profile_dir, 'archive.a')
    poplar_log_file = os.path.join(profile_dir, 'poplar_log.txt')
    poplibs_log_file = os.path.join(profile_dir, 'poplibs_log.txt')
    popart_log_file = os.path.join(profile_dir, 'popart_log.txt')
    stdout_file = os.path.join(profile_dir, 'stdout.txt')
    stderr_file = os.path.join(profile_dir, 'stderr.txt')

    tf_xla_dump_dir = os.path.join(profile_dir, "xla_dump")

    poplar_var_dump_file = os.path.join(profile_dir, 'vars.capnp')

    # Info files
    profile_info_file = os.path.join(profile_dir, 'profile_info.json')
    archive_info_file = os.path.join(profile_dir, 'archive_info.json')
    framework_file = os.path.join(profile_dir, 'framework.json')
    dot_file = os.path.join(profile_dir, 'graph.dot')
    graph_report_file = os.path.join(profile_dir, 'graph.json')
    execution_report_file = os.path.join(profile_dir, 'execution.json')
    tile_mappings = os.path.join(profile_dir, 'tile_mappings')

    # FIXME: TF requires the tile_mappings directory to exist and won't create it
    os.makedirs(tile_mappings, exist_ok=True)

    # Engine Options
    engine_options_new = {
        "target.saveArchive": archive_file,
        "debug.outputAllSymbols": "true",
        "debug.allowOutOfMemory": "true",
        "debug.loweredVarDumpFile": poplar_var_dump_file,
        # "debug.loweredVarDumpNumOOMTiles": "5"
    }
    if instrumentation:
        engine_options_new['debug.instrument'] = 'true'

    # Setup Environment Variables
    envs = dict(os.environ)
    #Avoid overwritting Environment Variable for the engine options
    if 'POPLAR_ENGINE_OPTIONS' in envs:
        engine_options=json.loads(envs['POPLAR_ENGINE_OPTIONS'])
        engine_options_new.update(engine_options)
        
    additional_envs = {
        'GC_PROFILE_LOG_DIR': profile_dir,
        'GC_PROFILE_PRETTY': str(pretty_json),

        'POPLAR_ENGINE_OPTIONS':json.dumps(engine_options_new),
        'POPLAR_LOG_LEVEL': 'DEBUG',
        'POPLAR_LOG_DEST': poplar_log_file,

        'POPLIBS_LOG_LEVEL': 'TRACE',
        'POPLIBS_LOG_DEST': poplibs_log_file,

        'TF_CPP_VMODULE': 'compiler=1,tensor=2',
        'TF_POPLAR_FLAGS': 
            f'--save_oom_profiler={os.path.join(profile_dir, "summary.txt")} '
            f'--tensor_map_file_path={tile_mappings}' ,
        'XLA_FLAGS':
            f'--xla_dump_to={tf_xla_dump_dir} '
            '--xla_dump_hlo_as_dot '
            '--xla_dump_hlo_pass_re=forward-allocation '
            '--xla_hlo_graph_sharding_color',

        'POPART_LOG_LEVEL': 'DEBUG',
        'POPART_LOG_DEST': popart_log_file,

        'PYTHONUNBUFFERED': 'TRUE'
    }
    envs.update(additional_envs)

    if verbose:
        print("Enviroment Variables:")
        for env, value in additional_envs.items():
            print(f" {env} : {value}")

    # Run program
    print(f"Running program (CTRL+C to exit early)\n\t{' '.join(program) if verbose else ''}\n")

    stdout = open(stdout_file, 'bw')
    stderr = open(stderr_file, 'bw')

    profile_info = {}
    profile_info["command"] = program
    profile_info["envs"] = additional_envs
    profile_info["start_time"] = time.time()
    proc = subprocess.Popen(
        program,
        env=envs,
        stdout=subprocess.PIPE if pipe_to_shell else stdout,
        stderr=stderr)

    try:
        if pipe_to_shell:
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                stdout.write(line)

        returncode = proc.wait()
    except KeyboardInterrupt:
        print("Ending program early...")
        returncode = proc.kill()
    profile_info["end_time"] = time.time()

    stdout.close()
    stderr.close()

    profile_info["return_code"] = returncode

    if returncode != 0:
        print("")
        warn(f"Program exited with code {returncode}. See {stderr_file} for more info.\n")

    print("Program Finished.")

    framework = detect_framework(stderr_file, popart_log_file)
    if verbose:
        print(f"Detected Framework: {framework}")

    postprocess(
        profile_dir,
        graph_report_file=graph_report_file,
        execution_report_file=execution_report_file,
        archive_file=archive_file,
        poplar_log_file=poplar_log_file,
        collect_sys_info=collect_sys_info,
        framework_file=framework_file,
        framework=framework,
        tf_tile_mappings=tile_mappings,
        tf_xla_dump=tf_xla_dump_dir,
        graph_dot_file=dot_file,
        profile_info=profile_info,
        name=name,
        verbose=verbose,
        hints=True,
        framework_checks=True
    )
    if profile_info["return_code"]:
        sys.exit(profile_info["return_code"])


class Files(object):
    archive = False
    graph = False
    execution = False
    profile = False
    svg = False
    tile_mapping = False

    def __iter__(self):
        return map(lambda attr: getattr(self, attr),
                   ['archive','graph','execution','profile','svg','tile_mapping'])

    def all(self):
        return reduce(lambda accl,attr: accl and attr, iter(self), True)

    def any(self):
        return reduce(lambda accl,attr: accl or attr, iter(self), False)

    def display_available_profiles(self):
        print("Available Profiles:\n")
        string = ""
        if self.profile:
            string += "Host Report (from profile_info.json)\n"
        if self.archive or self.graph:
            string += "Memory Report\n"
            max_tile = []
            if self.archive:
                max_tile.append("archive_info.json")
            if self.graph:
                max_tile.append("graph.json")
            string += f" Max Tile Memory (from {' & '.join(max_tile)})\n"
            if self.graph:
                string += " By Tile Memory Breakdown (from graph.json)\n"
        if self.graph:
            string += "Program Tree (from graph.json)\n"
        if self.execution:
            string += "Execution Trace (from execution.json)\n"
        if self.svg:
            string += "Graph Viewer (from graph.dot.svg)\n"
        if self.tile_mapping:
            string += "Tile Mapping (from tile_mapping.json)\n"
        print(string)

def postprocess(
    profile_dir,

    graph_report_file=None,
    execution_report_file=None,

    archive_file=None,
    archive_info_file=None,

    poplar_log_file=None,

    collect_sys_info=True,
    profile_info_file=None,

    framework=None,
    framework_file=None,

    tf_log_file=None,
    tf_xla_dump=None,
    tf_tile_mappings=None,
    tile_mapping_file=None,

    graph_dot_file=None,
    graph_svg_file=None,

    profile_info=None,
    name=None,

    verbose=False,
    hints=False,
    pretty_json=False,
    framework_checks=False
):
    results = Files()

    # Create profile directory
    profile_dir = os.path.abspath(profile_dir)
    if verbose:
        print(f"Making profile directory: {profile_dir}")
    os.makedirs(profile_dir, exist_ok=True)

    if profile_info is None:
        profile_info = {}

    # Profile Info
    if profile_info_file is None:
        if name is None:
            name = generate_slug()
            if verbose:
                print(f"Generated name: {name}")
        with open(os.path.join(sys.prefix, "gcprofile-data", "version.json")) as f:
            version = json.load(f)
        profile_info.update(dict(name=name, gcprofile_version="{}.{}.{}".format(version["major"], version["minor"], version["point"])))
        if collect_sys_info:
            if verbose:
                print("Collecting Host Information")
            profile_info["sys_info"] = gather_system_info()
    else:
        profile_info = json.load(profile_info_file)

    # Extract Framework information
    profile_info['framework'] = profile_info.get("framework", {})
    if (framework is None or framework == "unknown") and framework_file:
        try:
            with open(framework_file, 'r') as f:
                profile_info['framework'] = json.load(f)
        except FileNotFoundError:
            warn("Could not open framework file {}. {}"
                .format(framework_file,
                        "Are you using `gcprofile.save_x_report` to extract poplar reports?" if hints else ""))
    elif framework:
        profile_info['framework'] = { "framework": framework, "version": "Unknown" }

    # Tilemapping from tensorflow log (currently stderr until TF implements logging dest) -> tile_mapping.json
    if not framework_checks or profile_info['framework'].get("framework", None) == "tensorflow":
        # FIXME: 
        # Tile mapping format has changed. Uncomment this when tile mapping visualisation has been implemented
        # if tf_tile_mappings:
        #     tile_mapping_file = largest_tile_mapping_file(tf_tile_mappings)
        #     if tile_mapping_file and verbose:
        #         print(f"Chosen tile mapping {tile_mapping_file}")
        if tile_mapping_file is None:
            if tf_log_file is not None:
                if verbose:
                    print("Extracting tile_mapping from the tensorflow log.")
                mapping = extract_tf_tile_mapping(tf_log_file, pretty_json)
                if mapping:
                    with open(os.path.join(profile_dir, 'tile_mapping.json'), 'w') as f_:
                        f_.write(mapping)
                    results.tile_mapping = True
        else:
            copy(tile_mapping_file, os.path.join(profile_dir, 'tile_mapping.json'))
            results.tile_mapping = True

        if graph_svg_file is None:
            if tf_xla_dump:
                graph_dot_file = find_xla_dot_file(tf_xla_dump)
            if graph_dot_file:
                if verbose:
                    print("Converting the dot graph to SVG")
                try:
                    convert_graph_to_svg(graph_dot_file, os.path.join(profile_dir, 'graph.dot.svg'))
                    results.svg = True
                except Exception as e:
                    print(e)
                    warn("Could not convert dot file to svg. {}".format("Is graphviz installed?" if hints else ""))
        else:
            copy(graph_svg_file, os.path.join(profile_dir, 'graph.dot.svg'))
            results.svg = True


    # Archive Info.
    if archive_info_file is None:
        if archive_file:
            if verbose:
                print("Extracting information from archive file")
            try:
                max_tile_memory = extract_max_memory(archive_file)
                if len(max_tile_memory) > 0:
                    with open(os.path.join(profile_dir, 'archive_info.json'), 'w') as f:
                        json.dump(dict(
                            max_tile_memory=max_tile_memory
                        ), f, indent=2 if pretty_json else None)
                    results.archive = True
            except FileNotFoundError:
                warn('Could not open archive file {}. {}'
                     .format(archive_file,
                             "Did the program compile a graph?" if hints else ""))
    else:
        copy(archive_info_file, os.path.join(profile_dir, 'archive_info.json'))
        results.archive = True

    if graph_report_file:
        try:
            statinfo = os.stat(graph_report_file)
            if statinfo.st_size > 0:
                copy(graph_report_file, os.path.join(profile_dir, 'graph.json'))
                results.graph = True
        except FileNotFoundError:
            warn("Could not find the graph report file. {}".format(
                "Was the graph report saved? See README section: Retrieving Graph and Execution Reports." if hints else ""))
    if execution_report_file:
        try:
            statinfo = os.stat(execution_report_file)
            if statinfo.st_size > 0:
                copy(execution_report_file, os.path.join(profile_dir, 'execution.json'))
                results.execution = True
        except FileNotFoundError:
            warn("Could not find the execution report file. {}".format(
                "Was the execution report saved? See README section: Retrieving Graph and Execution Reports." if hints else ""))

    # Extract poplar information
    profile_info['poplar'] = profile_info.get("poplar", {})
    if poplar_log_file is not None:
        try:
            with open(poplar_log_file, 'r') as f:
                # Example to look for:
                # 15:53:49.625 48103 [I] Poplar version 0.8.17 (7b9aa6735e)
                # 15:53:49.625 48103 [D] Poplar package hash 3cc493e2a9
                poplar_regex = re.compile('Poplar version ([\d.]+)\s*\((\w+)\)')
                first = True
                for line in f:
                    match = poplar_regex.findall(line)
                    if match and first:
                        profile_info['poplar']['version'] = match[0][0]
                        profile_info['poplar']['hash'] = match[0][1]
                        poplar_regex = re.compile('Poplar package hash (\w+)')
                        first = False
                    elif match:
                        profile_info['poplar']['package_hash'] = match[0]
                        break
        except FileNotFoundError:
            fail("Could not open the poplar log file. {}".format("Did a poplar program run?" if hints else ""))

    if results.any():
        # Profile_info_file
        with open(os.path.join(profile_dir, 'profile_info.json'), 'w') as f:
            json.dump(profile_info, f, indent=2 if pretty_json else None)
            results.profile = True

        if verbose:
            results.display_available_profiles()
        finish(f"Profile finished. Files can be found: {profile_dir}")
    else:
        fail("Failed to create any profiling files.")
