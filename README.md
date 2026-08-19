# Pylontech FH3X protocol library

This package contains the Home Assistant-independent register definitions and
decoders for Pylontech Force H3X systems. It is shared by the HACS integration
and the Home Assistant Core integration so protocol behavior is implemented in
one place.

The package is intentionally limited to protocol data types and decoding. Home
Assistant lifecycle, entities, config flows, diagnostics, and discovery
transport adapters remain in their integrations.

