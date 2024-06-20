# Developer documentation

## UML Diagrams

Use a PlantUML previewer to generate Unified Modeling Language (UML) diagrams
from the PlantUML scripts (`*.puml`) in this subdirectory.
The generated versions below were created by the renderer on the
[PlantUML landing page](https://plantuml.com)

### Class Hierarchy

This diagram provides a high-level picture of the neural-fortran classes and
their interrelationships:
![class-hierarchy](https://user-images.githubusercontent.com/13108868/168928394-9fbf7880-0b11-4eb5-9106-baeb3ae3482d.png)

### Developer API

This diagram enhances the above class hierarchy depiction to include a richer
summation of the public interface of each class, including the public derived
types, type-bound procedures, and user-defined structure constructors:

![developer-api](https://user-images.githubusercontent.com/13108868/168961635-1f43641f-8144-4c4c-aa61-9f7140650e42.png)

For a depiction of the derived type components (but without the richness of
information on interrelationships), generate the HTML documentation using
[FORD](https://github.com/modern-fortran/neural-fortran#api-documentation).

### User API

This diagram depicts the functionality intended to be accessed directly by
neural-fortran users:

TBD
