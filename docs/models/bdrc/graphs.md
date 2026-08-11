# limnotech_rating_curves.models.bdrc.graphs

The marginal log-density as one pytensor graph, over shared dataset inputs.

## Why the dataset lives in shared variables
The algebra depends on the dataset only through a few arrays: the stages, the log
discharges, the variance weights, and for the gplm models the unique-stage indicator
and its distance matrix. Inlining those arrays into the graph as numpy constants makes
every distinct sample size a structurally different graph, which pytensor has to
optimize and compile again at a cost of about 40 seconds. A cross-validation fold
changes the sample size on every fold, so a batch run would spend hours recompiling the
same algebra.

The arrays therefore live in `SharedDataset`, and every shape is taken from them
symbolically (`stage.shape[0]`, never a Python `int`). The graph then encodes
nothing about the number of measurements, so one compiled function per
`(model, c known or inferred)` serves every site and every fold: eight graphs for the
whole package. Binding a dataset is `SharedDataset.bind`, which is a few
`set_value` calls.

The design pieces (l, varr, Sig_x, X) are defined once here and compiled for both uses,
gradients for the optimizer and the sampler and numeric values for the closed-form
posterior draws, so the algebra has a single definition.
