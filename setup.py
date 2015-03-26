from setuptools import setup

if __name__ == "__main__":
    setup(name="PyGRAFS",
      version="0.1",
      description="Gridded Atmospheric Forecast System",
      author="David John Gagne",
      author_email="dgagne@ucar.edu",
      packages=["pygrafs", "pygrafs.libs", "pygrafs.apps","pygrafs.scripts", "pygrafs.scripts.plotting"],
      install_requires=["numpy>=1.8", "pandas>=0.15", "scipy", "matplotlib", "netCDF4", "scikit-learn>=0.15", 'pyproj'])
