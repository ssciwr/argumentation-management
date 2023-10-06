# build to install treetagger dependency
import requests
import platform
import os
from pathlib import Path


DIRECTORY = ".treetagger"
TT_URL = "https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/"


def install_windows():
    filename = "tree-tagger-windows-3.2.3.zip"
    return filename


def install_linux():
    filename = "tree-tagger-linux-3.2.5.tar.gz"
    return filename


def install_macos():
    filename = "tree-tagger-MacOSX-Intel-3.2.3.tar.gz"
    return filename

if __name__ == "__main__":
    platform = platform.system()
    home = Path.home()
    Path(home / DIRECTORY).mkdir(parents=True, exist_ok=True)
    print("Installing treetagger into HOME/.treetagger...")
    if platform == "Windows":
        filename = install_windows()
    elif platform == "Linux":
        filename = install_linux()
    elif platform == "Darwin":
        filename = install_macos()
    else:
        raise OSError("Could not detect OS! Aborting build..")
    r = requests.get(TT_URL + filename, allow_redirects=True)
    open(home / DIRECTORY / filename, "wb").write(r.content)
    additional_files = [
        "tagger-scripts.tar.gz",
        "install-tagger.sh",
        "german.par.gz",
        "english.par.gz",
    ]
    for file in additional_files:
        r = requests.get(TT_URL + file, allow_redirects=True)
        open(home / DIRECTORY / file, "wb").write(r.content)
    # now go to treetagger dir and run install script
    os.chdir(home / DIRECTORY)
    os.system("sh install-tagger.sh")
    # append treetagger dir to bashrc?
    # here we would want to check if it is already there
    # also for windows this will not work
    # with open(home / ".bashrc", "a") as myfile:
    # myfile.write("TAGDIR={}\n".format(home / DIRECTORY ))
