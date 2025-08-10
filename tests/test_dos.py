from lieme.io import DOS
import os

def test_dos():
    root = os.getcwd()
    os.chdir(root + "/tests/test_dos/test_1")
    dos = DOS()
    temp_dos_up, _ = dos.get_orbital_projected_dos("s")
    s_band_center = dos.get_band_center(temp_dos_up)
    assert isinstance(s_band_center, float)
    temp_dos_up, _ = dos.get_orbital_projected_dos("p")
    p_band_center = dos.get_band_center(temp_dos_up)
    assert isinstance(p_band_center, float)
    temp_dos_up, _ = dos.get_orbital_projected_dos("d")
    d_band_center = dos.get_band_center(temp_dos_up)
    assert isinstance(d_band_center, float)
    temp_dos_up, _ = dos.get_orbital_projected_dos("f")
    f_band_center = dos.get_band_center(temp_dos_up)
    assert isinstance(f_band_center, float)
    try:
        dos_up, _ = [0]*len(dos.energies), None
        dos.get_band_center(dos_up)
    except ZeroDivisionError:
        pass
    os.chdir(root + "/tests/test_dos/test_2")
    dos = DOS()
    temp_dos_up, temp_dos_down = dos.get_orbital_projected_dos("s")
    s_band_center = dos.get_band_center(temp_dos_up, temp_dos_down)
    assert isinstance(s_band_center, float)
    temp_dos_up, temp_dos_down = dos.get_orbital_projected_dos("f")
    f_band_center = dos.get_band_center(temp_dos_up, temp_dos_down)
    f_band_center = 0.0 if f_band_center is 0 else f_band_center
    assert isinstance(f_band_center, float)
    os.chdir(root)