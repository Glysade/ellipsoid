# Ellipsoid Background

## Ellipsoids

### Theory of Ellipsoids

An ellipsoid is a solid three dimensional version of an ellipse.
The points inside an ellipse can be described in quadratic form by a 3x3 square matrix, or in parametric 
form using the directions and magnitudes of the principal axes.

* [General information in Wikipedia](https://en.wikipedia.org/wiki/Ellipsoid)
* [Quadratic forms and ellipsoids](https://laurentlessard.com/teaching/cs524/slides/11%20-%20quadratic%20forms%20and%20ellipsoids.pdf)

### MVEE

MVEE is the minimum volume enclosing ellipsoid. Given a set of points we can calculate the smallest 
ellipsoid that contains those points.

This is interesting for drug discovery as:

* Drug molecules have three dimensional shapes that fit inside the proteins they bind to. E.g. [HIV protease](https://www.rcsb.org/3d-view/2P3B)
* If we have one good drug (an active compound) another compound may also be active if it has the same shape
* Comparing the shapes of molecules can be difficult but comparing ellipsoids instead could be easier

Methods to calculate the MVEE from a set of points can be found online

* [Theory](https://citeseerx.ist.psu.edu/doc/10.1.1.116.7691)
* [Another link to the above](https://citeseerx.ist.psu.edu/doc/10.1.1.116.7691)
* [Python code to calculate MVEE and ellipse example](https://gist.github.com/Gabriel-p/4ddd31422a88e7cdf953)
* [Matlab MVEE](https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-elipsoid)
* Stackoverflow [port Matlab MVEE to Python](https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python)
* Stackoverflow [calculate MVEE in Java](https://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440)

The MVEE code is downloaded to [get_ellipse.py](../downloaded_code/get_ellipse.py) and you can run it to calculate a MVEE from some random 2D points.

The MVEE code returns the quadratic square matrix that defines the MVEE.  Decomposing that matrix using Singular Value Decomposition (`SVD`) yields the eigen values and eigen vectors of the quadratic matrix and from those the axes vectors of the ellipsoid may be determined.
  
### Programming chemistry

When storing molecules in computer files we can use

* `Smiles` [strings](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)
* `Mol` block records, [summary](https://en.wikipedia.org/wiki/Chemical_table_file)
  and [full documentation](https://discover.3ds.com/sites/default/files/2020-08/biovia_ctfileformats_2020.pdf)

Smiles strings do not include molecular coordinates but mol block records can.

We use RDKit to program molecules

* [Getting started in Python](https://www.rdkit.org/docs/GettingStartedInPython.html)
* [RDKit book](https://www.rdkit.org/docs/RDKit_Book.html)
* [Python API](https://www.rdkit.org/docs/api-docs.html)
* [Github site](https://github.com/rdkit/rdkit)

Programming in Python is much easier with an IDE. [Visual studio code](https://code.visualstudio.com/) is a great IDE for developing in [Python](https://code.visualstudio.com/docs/python/python-tutorial)

### Viewing molecules

There are a bunch of programs for viewing and editing molecules.  [PYMol](https://en.wikipedia.org/wiki/PyMOL) is a nice program for viewing molecules with a Python API.

The free version has a [manual install](https://pymolwiki.org/index.php/Windows_Install).

The PyMOL GUI can run Python scripts.  See the [manual](https://pymol.sourceforge.net/newman/userman.pdf)

This [PyMOL script](https://rbvi.github.io/chimerax-recipes/ellipsoid/ELM_ellipsoid.py) can display an
ellipsoid in Python.  The file is downloaded to [ELM_ellipsoid.py](../downloaded_code/ELM_ellipsoid.py)

### Getting started

I have written some example code in [Molecule.py](../Molecule.py). This:

* Starts with an example molecule defined by a smiles string
* Calculates 3D coordinates using RDKit (a molecular *conformation*)
* Finds the MVEE that contains the 3D coordinates
* Uses SVD to determine the ellipse axes
* Writes out the 3D molecules as a Mol block file
* Writes out a PyMOL script that will load the molecule and display the MVEE

Your computer already has 

* Visual studio code
* Git bash
* Python
* Pymol

You will need to install

* Windows terminal
* Visual studio code extensions

We will need to spend some time making sure that everything works with your login.

### Things to do next

* Use [Toggl](https://toggl.com/) to track time
* Document using Markdown (as in this document)
* Run the Python code from the command line and load the result into PyMol
* Get familiar with the Visual studio code IDE
* Explore using the debugger
* Change the code so that you can pass any smile as a command line argument.  Create ellipsoids for 
  different structures
* Use the PyMol API to add the ellipsoid axes to visualization (see `Compiled Graphics Objects` in the 
  PyMol manual)
* Use [unit tests](https://code.visualstudio.com/docs/python/testing) so you can define and document
  your program's behavior.
* What happens when you try to create an ellipsoid for a "flat" structure like benzene?  I guess 
  it will crash.  What can be done to fix that. 
* More advanced- use RDKit to find all the rings in a molecule.  Chop the molecule up into rings 
  and the bits between them (cyclic and acyclic structures).  Instead of one ellipse for a molecule create multiple ellipses. Issues:
  * Fused rings
  * Atoms adjacent to rings
  * Do some atoms need to be in more than one ellipse?