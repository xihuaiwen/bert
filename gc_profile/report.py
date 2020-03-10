import os
import json

LOG_DIR = os.environ.get('GC_PROFILE_LOG_DIR', None)
PRETTY = os.environ.get('GC_PROFILE_PRETTY', False)

def is_popart_session(popart, session):
    return (hasattr(popart, 'Session') and isinstance(session, popart.Session)) or \
            isinstance(session, popart.InferenceSession) or \
            isinstance(session, popart.TrainingSession)

def is_popart_prepare_device_exception(popart, exception):
    return (hasattr(popart, 'PrepareDeviceException') and isinstance(exception, popart.PrepareDeviceException))

def save_popart_report(session, log_dir=LOG_DIR, exception=None):
    if log_dir is None:
        print("[gc_profile]: Could not find GC_PROFILE_LOG_DIR env var. Run with `gc_profile -- PROGRAM`")
        return

    try:
        try:
            import popart
        except ImportError:
            import poponnx as popart
    except ImportError as e:
        raise ImportError('Could not import popart.'
                          ' Are you profiling a popart program?'
                          ' If not, try `save_tf_profile` instead.')

    assert (is_popart_session(popart, session) == True), 'Argument session must be either a InferenceSession or TrainingSession'

    # Save the framework information
    with open(os.path.join(log_dir, 'framework.json'), 'w') as f:
        json.dump(dict(
            framework='popart',
            version=popart.__version__,
        ), f, separators=(',', ':'))

    if (exception is not None) and (is_popart_prepare_device_exception(popart, exception)):
        # Save the graph report
        with open(os.path.join(log_dir, "graph.json"), "wb") as f:
            graph_report = exception.getGraphReport()
            f.write(graph_report)

        with open(os.path.join(log_dir, "summary.log"), "w") as f:
            summary_report = exception.getSummaryReport()
            f.write(summary_report)

        execution_report = None

    else:
        # Save the graph & execution reports
        with open(os.path.join(log_dir, "graph.json"), "wb") as f:
            graph_report = session.getGraphReport()
            f.write(graph_report)

        # with open(os.path.join(log_dir, "summary.log"), "w") as f:
        #     summary_report = session.getSummaryReport()
        #     f.write(summary_report)

        with open(os.path.join(log_dir, "execution.json"), "wb") as f:
            execution_report = session.getExecutionReport()
            f.write(execution_report)

        # Save the serialized graph
        with open(os.path.join(log_dir, "serialized_graph.json"), 'wb') as f:
            serialized_graph = session.getSerializedGraph()
            f.write(serialized_graph)

    # Save the tile mapping
    with open(os.path.join(log_dir, "ptile_mapping.json"), 'w') as f:
        json.dump(session.getTensorTileMap(), f, separators=(',', ':'))

    return {"graph": graph_report, "execution": execution_report}




def save_tf_report(event_trace_protos, log_dir=LOG_DIR):
    if log_dir is None:
        print("[gc_profile]: Could not find GC_PROFILE_LOG_DIR env var. Run with `gc_profile -- PROGRAM`")
        return

    try:
        import tensorflow as tf
        from tensorflow.python.ipu import utils
        from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
    except ImportError as e:
        raise ImportError('Could not import tensorflow.'
                          ' Are you profiling a tensorflow program?'
                          ' If not, try `save_popart_profile` instead.')


    assert hasattr(type(event_trace_protos), '__iter__'), \
        ('Argument event_trace_protos must be the '
        'list returned from session.run(report).\n'
        '\t\twhere report = tf.compiler.plugin.poplar.ops.gen_ipu_ops.ipu_event_trace().')

    # Check if any events are present.
    if len(event_trace_protos) == 0:
        print("[gc_profile]: No IPUEventTrace recorded. Make sure to run save_tf_report after "
              "the program's session.run.")
        return

    # Save the reports
    events = list(map(IpuTraceEvent.FromString, reversed(event_trace_protos)))
    largest_graph_idx = None
    largest_exec_idx = None
    for idx, event in enumerate(events):
        if event.type == IpuTraceEvent.COMPILE_END and len(event.compile_end.compilation_report) > 0 and \
            (largest_graph_idx is None or len(event.compile_end.compilation_report) > len(events[largest_graph_idx].compile_end.compilation_report)):
            largest_graph_idx = idx
        if event.type == IpuTraceEvent.EXECUTE and len(event.execute.execution_report) > 0 and \
            (largest_exec_idx is None or len(event.execute.execution_report) > len(events[largest_exec_idx].execute.execution_report)):
            largest_exec_idx = idx

    if largest_graph_idx is not None:
        with open(os.path.join(log_dir, "graph.json"), "w") as f:
            graph_report = events[largest_graph_idx].compile_end.compilation_report.decode('utf-8')
            f.write(graph_report)
    else:
        graph_report = None
        print("[gc_profile]: Could not find a graph report. Use `profiling=True` when calling `create_ipu_config`")

    if largest_exec_idx is not None:
        with open(os.path.join(log_dir, "execution.json"), "w") as f:
            execution_report = events[largest_exec_idx].execute.execution_report.decode('utf-8')
            f.write(execution_report)
    else:
        execution_report = None
        print("[gc_profile]: Could not find an execution report. Use `profile_execution=True` when calling `create_ipu_config`")

    # Save the framework information
    with open(os.path.join(log_dir, 'framework.json'), 'w') as f:
        json.dump(dict(
            framework='tensorflow',
            version=tf.__version__,
        ), f, separators=(',', ':'))

    return {"graph": graph_report, "execution": execution_report}

save_poponnx_report = save_popart_report

