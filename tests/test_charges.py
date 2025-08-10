from lieme.io import get_atoms_with_charges
import os

def test_get_atoms_with_charges():
    root = os.getcwd()
    os.chdir(root + "/tests/test_charges/test_1")
    get_atoms_with_charges()
    os.chdir(root + "/tests/test_charges/test_2")
    get_atoms_with_charges()
    os.chdir(root + "/tests/test_charges/test_3")
    try:
        get_atoms_with_charges()
    except IOError:
        pass
    os.chdir(root + "/tests/test_charges/test_4")
    try:
        get_atoms_with_charges()
    except IOError:
        pass
    os.chdir(root + "/tests/test_charges/test_5")
    try:
        get_atoms_with_charges()
    except IOError:
        pass
    os.chdir(root)