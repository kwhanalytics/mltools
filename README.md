# mltools
`mltools` is an open source library to support reproducible, transparent machine learning selection and evaluation frameworks. This repository is made public as part of a research project funded by a grant from the Department of Energy's Solar Energy Technologies Office.

To run `mltools`, you can run:
```
python mltools.py --config_path './example/config/config_example.yaml' --iterations 5
```

`mltools` uses configuration yaml files to determine what data to use, what models to train, and what types of reports to write. An example configuration file can be located in `mltools/example/config` and can be used as a base template for building your own machine learning frameworks.