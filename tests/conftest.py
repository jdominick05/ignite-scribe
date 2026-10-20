# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests never read the user's own policy file (``ignite-scribe setup`` writes one to %LOCALAPPDATA%)."""
import os

os.environ.setdefault("IGNITE_SCRIBE_POLICY", "")
