Documentation
=============

UML Diagrams
------------
Use a PlantUML previewer to generate Unified Modeling Language (UML) diagrams from the PlantUML scripts (`*.puml`) in this subdirectory.  The generated versions below were created by in the renderer on the landing page at [PlantUML.com]

### Class Hierarchy
This diagram provides a high-level picture of the neural-fortran classes and their interrelatioships:
![class-hierarchy](https://user-images.githubusercontent.com/13108868/168928394-9fbf7880-0b11-4eb5-9106-baeb3ae3482d.png)

### Developer API
This diagram enhances the above class hierarchy depiction to include a richer summation of thepublic interface of  each class, including the public derived types, type-bound procecures, and user-defined structure constructors:
![developer-api](https://user-images.githubusercontent.com/13108868/168961635-1f43641f-8144-4c4c-aa61-9f7140650e42.png)
For a depection of the derived type components (but without the richness of information on interrelationships), see the HTML documentation.

### User API
This diagram depicts the functionality intended to be accessed directly by neural-fortran users:

TBD

HTML Pages
----------

With [FORD] installed, running the following command produces a doc/html folder containing additional nerual-fortran documentation:
```
ford ford.md
```

[FORD]: https://github.com/Fortran-FOSS-Programmers/ford/
[PlantUML]: https://plantuml.com
[Atom]: https://atom.io
