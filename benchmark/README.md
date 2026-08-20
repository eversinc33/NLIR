# NLIR benchmark corpus

This corpus has safe synthetic prompt text. It does not run the text.

Each family has one direct instruction and six context variants. The variants use negation, description, policy text, a quote, a hypothetical case, and an attacker description.

The IR expectations test text meaning. Rule expectations test hunting hypotheses. A rule hit on a context variant is allowed unless the case explicitly forbids that hit.

All URLs use the reserved `.invalid` suffix. All secrets and package names are placeholders.
