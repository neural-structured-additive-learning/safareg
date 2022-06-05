# safareg: Structured Additive Factorization Regression

[![R build status](https://github.com/neural-structured-additive-learning/safareg/workflows/R-CMD-check/badge.svg)](https://github.com/neural-structured-additive-learning/safareg/actions)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

Add-on pacakge for `deepregression` to fit structured additive models when the number of factor levels is prohibitively large and needs an efficient implementation, when fitting varying coefficient terms with a factorization approach, factorized effects in general, (higher-order) factorization machines, or additive (higher-order) factorization machines.

## Requirements

See [those of `deepregression`](https://github.com/neural-structured-additive-learning/deepregression/blob/main/README.md).

## Usage

In (one of) your formula(s) in `deepregression` you can use any function (e.g., `fm`) and pass the respective processor (e.g., `hofm_processor`) via the `additional_processors` argument in `deepregression`. For example:

```
mod <- deepregression(
  y = y,
  list_of_formulas = list(~ 1 + age + fm(V1, V2, V3, V4, V5, V6)),
  data = data,
  family = "bernoulli", 
  additional_processors = list(fm = hofm_processor)
)
```

The following processors are available:

* `fac_processor`: efficient computation of categorical effects with many factor levels
* `interaction_processor`: same as `fac_processor` but for interactions of two categorical effects
* `vc_processor`: efficient computation of varying coefficients (interaction of smooth and one or two categorical effect)
* `am_processor`: same as `vc_processor` but using an linear array model-type formulation (only for one categorical effect)
* `fz_processor`: computes a matrix factorization for two or three categorical effect interactions
* `vf_processor`: same as `vc_processor` with two levels, but using a factorization approach as in the `fz_processor` 
* `hofm_processor`: computes (higher-order) factorization machines for a given set of features
* `afm_processor`: computes additive factorization machines for a given set of features
* `ahofm_processor`: computes additive higher-order factorization machines for a given set of features

## Citation

When using or referencing the contents of this package, cite

    @article{FaStR,
      title={Factorized Structured Regression for Large-Scale Varying Coefficient Models},
      author={David R{\"u}gamer and Andreas Bender and Simon Wiegrebe and Daniel Racek and Bernd Bischl and Christian M{\"u}ller and Clemens Stachl},
      year={2022},
      journal = {arXiv preprint arXiv:2205.13080}
    }
    
for the efficient factor effect implementation or (time-varying) factorization approaches and

    @article{AFM,
      title = {Additive Higher-Order Factorization Machines},
      author = {David R\"ugamer},
      year = 2022,
      journal={arXiv preprint arXiv:2205.14515}
    }
    
for (higher-order) factorization machines or additive (higher-order) factorization machines.
