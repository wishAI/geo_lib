# Stellaris 4.4.6 Safe Load Patch

This is a conservative compatibility patch for the installed 4.4.6 playset.
It fixes EQN-GO's obsolete sequential-name localisation variables and disables
two unsafe ACOT/SBTG bridge paths that are recorded in the local crash logs.

It does **not** repair a save that has already been loaded under Stellaris
4.5.0, a save containing an invalid `CDeadCosmicStorm`, or malformed saved
ship/starbase objects. Use a save from before the version-mixed launches.

## Install

Copy this directory to `~/.local/share/Paradox Interactive/Stellaris/mod/`
as `stellaris_446_safe_load`. Create a launcher descriptor pointing to that
directory, add the mod to the playset, and place it last—after ACOT, SBTG,
ACOT Override, all translations, and EQN-GO.

Use it only with Stellaris 4.4.6 and a 4.4-compatible version of every mod.
Do not combine it with the installed Gigastructural Engineering or UI Overhaul
Dynamic builds that advertise 4.5 support.
