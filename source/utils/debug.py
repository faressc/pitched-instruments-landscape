import debugpy
debugpy.listen(("localhost", 5679))
debugpy.wait_for_client()