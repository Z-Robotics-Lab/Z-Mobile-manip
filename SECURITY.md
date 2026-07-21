# Security

Z-Mobile-Manip can command a mobile base, robot arm, and gripper. Treat every motion-capable
deployment as a supervised laboratory system.

- Keep the web UI on loopback or behind an authenticated operator-controlled tunnel.
- Keep a physical emergency stop within reach and maintain a clear operating area.
- Never commit `.env`, SSH keys, robot-specific Home poses, calibration files, rosbag data,
  captured RGB-D images, runtime logs, model licenses, or machine-bound activation data.
- Rotate any credential that is accidentally published before removing it from Git history.

Report security issues privately through this repository's GitHub Security Advisory page. Include
the affected commit, deployment topology, and a minimal reproduction that does not contain secrets
or unsafe actuator instructions.
