"""Hardware abstraction layer.

Every hardware-touching capability lives behind one function/class
signature here, with a "mac" implementation (usable today, on the dev
laptop) and an "rpi" implementation (fills in once the Pi hardware is on
hand). Selection is via vcm.config.get_platform() — nothing outside this
package should branch on platform directly.
"""
