"""fs_launch — THE ONLY launch path (/fs-launch): a confirmed fs_launch_spec -> a launch_record.

FsLaunchSkill composes confirm_before_launch and then calls
``foundationskills.interfaces.fs.launch.launch``; it re-shows the recomputed plan hash and never
fills it in (FL-IN-001/002/003) and refuses to call a local run a pass (FL-HO-001).
"""
from foundationskills.skills.fs_launch.skill import FsLaunchSkill

__all__ = ("FsLaunchSkill",)
