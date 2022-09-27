Documentation
=============

UML Diagrams
------------
Use a PlantUML previewer to generate Unified Modeling Language (UML) diagrams from the PlantUML scripts (the `*.puml` files) in this subdirectory.  The generated versions below were created by the [Online Server](https://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000) at [PlantUML.com](https://www.plantuml.com).

### Class Hierarchy
This diagram provides a high-level picture of the neural-fortran classes and their interrelatioships:
![class-hierarchy](https://user-images.githubusercontent.com/13108868/168928394-9fbf7880-0b11-4eb5-9106-baeb3ae3482d.png)

### Developer API
This diagram enhances the above class hierarchy depiction to include a richer summation of the public interface of  each class, including the public derived types, type-bound procecures, and user-defined structure constructors:
![developer-api](https://user-images.githubusercontent.com/13108868/168961635-1f43641f-8144-4c4c-aa61-9f7140650e42.png)
For a depiction of the derived type components (but without the richness of information on interrelationships), see the `ford`-generated HTML documentation on the neural-fortran GitHub Pages [site](https://modern-fortran.github.io/neural-fortran/).

### User API
This diagram depicts the functionality intended to be accessed directly by neural-fortran users:

**To Do**
Document the following types and/or procedures from the `nf` module:
- [] label_digits, load_mnist
- [] layer
- [] conv2d, dense, flatten, input, maxpool2d
- [] network
- [] sgd

HTML Pages
----------

With [ford] installed, running the following command produces a doc/html folder containing additional nerual-fortran documentation:
```
ford ford.md
```

[ford]: https://github.com/Fortran-FOSS-Programmers/ford/
[PlantUML]: https://plantuml.com
[Atom]: https://atom.io
