import os

if os.getenv("DEBUG_MODE") == "1" and os.getenv("DEBUG_SESSION_STARTED") != "1":
    import debugpy
    debugpy.listen(("localhost", 8080))
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()
    print("Debug mode is enabled.")

    os.environ["DEBUG_SESSION_STARTED"] = "1"
