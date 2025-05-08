import re


LOG_FOLDER = "/tmp/infscale"
PRINT_COLOR = {"success": "\033[92m", "failed": "\033[91m", "black": "\033[0m"}
ERROR_PATTERN = re.compile(
    r"(Traceback|AssertionError|Exception|InfscaleException|FAILED|Bad Request)"
)
