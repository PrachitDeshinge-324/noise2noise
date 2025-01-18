# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Submit a function to be run either locally or in a computing cluster."""

import copy
import io
import os
import pathlib
import pickle
import platform
import pprint
import re
import shutil
import time
import traceback
import typeguard

import zipfile

from enum import Enum

from .. import util
from ..util import EasyDict


class SubmitTarget(Enum):
    """The target where the function should be run.

    LOCAL: Run it locally.
    """
    LOCAL = 1


class PathType(Enum):
    """Determines in which format should a path be formatted.

    WINDOWS: Format with Windows style.
    LINUX: Format with Linux/Posix style.
    AUTO: Use current OS type to select either WINDOWS or LINUX.
    """
    WINDOWS = 1
    LINUX = 2
    AUTO = 3


_user_name_override = None


class SubmitConfig(util.EasyDict):
    """Strongly typed config dict needed to submit runs.

    Attributes:
        run_dir_root: Path to the run dir root. Can be optionally templated with tags. Needs to always be run through get_path_from_template.
        run_desc: Description of the run. Will be used in the run dir and task name.
        run_dir_ignore: List of file patterns used to ignore files when copying files to the run dir.
        run_dir_extra_files: List of (abs_path, rel_path) tuples of file paths. rel_path root will be the src directory inside the run dir.
        submit_target: Submit target enum value. Used to select where the run is actually launched.
        num_gpus: Number of GPUs used/requested for the run.
        print_info: Whether to print debug information when submitting.
        ask_confirmation: Whether to ask a confirmation before submitting.
        use_typeguard: Whether to use the typeguard module for run-time type checking (slow!).
        run_id: Automatically populated value during submit.
        run_name: Automatically populated value during submit.
        run_dir: Automatically populated value during submit.
        run_func_name: Automatically populated value during submit.
        run_func_kwargs: Automatically populated value during submit.
        user_name: Automatically populated value during submit. Can be set by the user which will then override the automatic value.
        task_name: Automatically populated value during submit.
        host_name: Automatically populated value during submit.
    """

    def __init__(self):
        super().__init__()

        # run (set these)
        self.run_dir_root = ""  # should always be passed through get_path_from_template
        self.run_desc = ""
        self.run_dir_ignore = ["__pycache__", "*.pyproj", "*.sln", "*.suo", ".cache", ".idea", ".vs", ".vscode"]
        self.run_dir_extra_files = None

        # submit (set these)
        self.submit_target = SubmitTarget.LOCAL
        self.num_gpus = 1
        self.print_info = False
        self.ask_confirmation = False
        self.use_typeguard = False

        # (automatically populated)
        self.run_id = None
        self.run_name = None
        self.run_dir = None
        self.run_func_name = None
        self.run_func_kwargs = None
        self.user_name = None
        self.task_name = None
        self.host_name = "localhost"


def get_path_from_template(path_template: str) -> str:
    """Convert a path template to an actual path."""
    if not path_template:
        return ''
    
    # Always use absolute paths
    path = os.path.abspath(path_template)
    
    # Use os.path.normpath for cross-platform compatibility
    return os.path.normpath(path)


def get_template_from_path(path: str) -> str:
    """Convert a normal path back to its template representation."""
    # replace all path parts with the template tags
    path = path.replace("\\", "/")
    return path


def convert_path(path: str, path_type: PathType = PathType.AUTO) -> str:
    """Convert a normal path to template and the convert it back to a normal path with given path type."""
    path_template = get_template_from_path(path)
    path = get_path_from_template(path_template)
    return path


def set_user_name_override(name: str) -> None:
    """Set the global username override value."""
    global _user_name_override
    _user_name_override = name


def get_user_name():
    """Get the current user name."""
    if _user_name_override is not None:
        return _user_name_override
    
    system = platform.system().lower()
    try:
        if system == "windows":
            return os.getlogin()
        elif system in ["linux", "darwin"]:  # Handle both Linux and macOS
            import pwd
            return pwd.getpwuid(os.geteuid()).pw_name
        else:
            # Fallback to a generic username if platform detection fails
            return os.getenv('USER', 'unknown_user')
    except:
        # Last resort fallback
        return 'unknown_user'


def _create_run_dir_local(submit_config: SubmitConfig) -> str:
    """Create a new run dir with increasing ID number at the start."""
    run_dir_root = get_path_from_template(submit_config.run_dir_root)

    if not os.path.exists(run_dir_root):
        print("Creating the run dir root: {}".format(run_dir_root))
        os.makedirs(run_dir_root)

    submit_config.run_id = _get_next_run_id_local(run_dir_root)
    submit_config.run_name = "{0:05d}-{1}".format(submit_config.run_id, submit_config.run_desc)
    run_dir = os.path.join(run_dir_root, submit_config.run_name)

    if os.path.exists(run_dir):
        raise RuntimeError("The run dir already exists! ({0})".format(run_dir))

    print("Creating the run dir: {}".format(run_dir))
    os.makedirs(run_dir)

    return run_dir


def _get_next_run_id_local(run_dir_root: str) -> int:
    """Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]
    r = re.compile("^\\d+")  # match one or more digits at the start of the string
    run_id = 0

    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)

    return run_id


def _populate_run_dir(run_dir: str, submit_config: SubmitConfig) -> None:
    """Copy all necessary files into the run dir. Assumes that the dir exists, is local, and is writable."""
    print("Copying files to the run dir")
    files = []

    run_func_module_dir_path = util.get_module_dir_by_obj_name(submit_config.run_func_name)
    assert '.' in submit_config.run_func_name
    for _idx in range(submit_config.run_func_name.count('.') - 1):
        run_func_module_dir_path = os.path.dirname(run_func_module_dir_path)
    files += util.list_dir_recursively_with_ignore(run_func_module_dir_path, ignores=submit_config.run_dir_ignore, add_base_to_relative=False)

    dnnlib_module_dir_path = util.get_module_dir_by_obj_name("dnnlib")
    files += util.list_dir_recursively_with_ignore(dnnlib_module_dir_path, ignores=submit_config.run_dir_ignore, add_base_to_relative=True)

    if submit_config.run_dir_extra_files is not None:
        files += submit_config.run_dir_extra_files

    files = [(f[0], os.path.join(run_dir, "src", f[1])) for f in files]
    files += [(os.path.join(dnnlib_module_dir_path, "submission", "_internal", "run.py"), os.path.join(run_dir, "run.py"))]

    util.copy_files_and_create_dirs(files)

    pickle.dump(submit_config, open(os.path.join(run_dir, "submit_config.pkl"), "wb"))

    with open(os.path.join(run_dir, "submit_config.txt"), "w") as f:
        pprint.pprint(submit_config, stream=f, indent=4, width=200, compact=False)


def run_wrapper(submit_config: SubmitConfig) -> None:
    """Wrap the actual run function call for handling logging, exceptions, typing, etc."""
    is_local = submit_config.submit_target == SubmitTarget.LOCAL

    checker = None

    if submit_config.use_typeguard:
        checker = typeguard.TypeChecker("dnnlib")
        checker.start()

    # when running locally, redirect stderr to stdout, log stdout to a file, and force flushing
    if is_local:
        logger = util.Logger(file_name=os.path.join(submit_config.run_dir, "log.txt"), file_mode="w", should_flush=True)
    else:  # when running in a cluster, redirect stderr to stdout, and just force flushing (log writing is handled by run.sh)
        logger = util.Logger(file_name=None, should_flush=True)

    import dnnlib
    dnnlib.submit_config = submit_config

    try:
        print("dnnlib: Running {0}() on {1}...".format(submit_config.run_func_name, submit_config.host_name))
        start_time = time.time()
        util.call_func_by_name(func_name=submit_config.run_func_name, submit_config=submit_config, **submit_config.run_func_kwargs)
        print("dnnlib: Finished {0}() in {1}.".format(submit_config.run_func_name, util.format_time(time.time() - start_time)))
    except:
        if is_local:
            raise
        else:
            traceback.print_exc()

            log_src = os.path.join(submit_config.run_dir, "log.txt")
            log_dst = os.path.join(get_path_from_template(submit_config.run_dir_root), "{0}-error.txt".format(submit_config.run_name))
            shutil.copyfile(log_src, log_dst)
    finally:
        open(os.path.join(submit_config.run_dir, "_finished.txt"), "w").close()

    dnnlib.submit_config = None
    logger.close()

    if checker is not None:
        checker.stop()


def submit_run(submit_config: SubmitConfig, run_func_name: str, **run_func_kwargs) -> None:
    """Submit a run with improved error handling."""
    submit_config = copy.copy(submit_config)
    
    # Set username with better error handling
    if submit_config.user_name is None:
        submit_config.user_name = get_user_name()
    
    submit_config.run_func_name = run_func_name
    submit_config.run_func_kwargs = run_func_kwargs

    # Ensure run_dir exists and is absolute
    if submit_config.run_dir_root:
        submit_config.run_dir_root = os.path.abspath(submit_config.run_dir_root)
        os.makedirs(submit_config.run_dir_root, exist_ok=True)
        
    if submit_config.run_dir:
        submit_config.run_dir = os.path.abspath(submit_config.run_dir)
        os.makedirs(submit_config.run_dir, exist_ok=True)

    assert submit_config.submit_target == SubmitTarget.LOCAL
    if submit_config.submit_target in {SubmitTarget.LOCAL}:
        run_dir = _create_run_dir_local(submit_config)
        submit_config.task_name = "{0}-{1:05d}-{2}".format(submit_config.user_name, submit_config.run_id, submit_config.run_desc)
        submit_config.run_dir = run_dir
        _populate_run_dir(run_dir, submit_config)

    if submit_config.print_info:
        print("\nSubmit config:\n")
        pprint.pprint(submit_config, indent=4, width=200, compact=False)
        print()

    if submit_config.ask_confirmation:
        if not util.ask_yes_no("Continue submitting the job?"):
            return

    run_wrapper(submit_config)
