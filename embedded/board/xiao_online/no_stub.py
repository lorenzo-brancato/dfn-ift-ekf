# no_stub.py — PlatformIO extra script: flash without the esptool stub loader.
# On this PC (Windows, XIAO ESP32-S3 native USB) the stub stops answering right
# after it starts ("No serial data received"); the ROM loader works.
# --no-stub is a global esptool option, so it must go before "write_flash",
# where upload_flags cannot put it.
Import("env")  # noqa: F821  (provided by SCons)

flags = list(env["UPLOADERFLAGS"])  # noqa: F821
if "write_flash" in flags and "--no-stub" not in flags:
    k = flags.index("write_flash")
    env.Replace(UPLOADERFLAGS=flags[:k] + ["--no-stub"] + flags[k:])  # noqa: F821
