# safareg: Structured Additive Factorization Regression

[![R build status](https://github.com/neural-structured-additive-learning/safareg/workflows/R-CMD-check/badge.svg)](https://github.com/neural-structured-additive-learning/safareg/actions)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Add-on pacakge for the R package [`deepregression`](https://github.com/neural-structured-additive-learning/deepregression) to fit structured additive models when the number of factor levels is prohibitively large and needs an efficient implementation, when fitting varying coefficient terms with a factorization approach, factorized effects in general, (higher-order) factorization machines, or additive (higher-order) factorization machines.

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

    @InProceedings{FaStR,
      title={Factorized Structured Regression for Large-Scale Varying Coefficient Models},
      author={David R{\"u}gamer and Andreas Bender and Simon Wiegrebe and Daniel Racek and Bernd Bischl and Christian M{\"u}ller and Clemens Stachl},
      year={2023},
      publisher={Springer International Publishing},
      booktitle="Machine Learning and Knowledge Discovery in Databases",
      publisher="Springer Nature Switzerland",
      address="Cham",
      pages="20--35"
    }
    
for the efficient factor effect implementation or (time-varying) factorization approaches and

    @InProceedings{AFM,
      title = {Scalable Higher-Order Tensor Product Spline Models},
      author = {David R\"ugamer},
      year={2024},
      booktitle = 	 {Proceedings of The 27th International Conference on Artificial Intelligence and Statistics},
      series = 	 {Proceedings of Machine Learning Research},
      publisher =    {PMLR}
}
    
for (higher-order) factorization machines or additive (higher-order) factorization machines.
