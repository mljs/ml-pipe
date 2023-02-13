# Changelog

## [0.3.0](https://github.com/mljs/ml-pipe/compare/v0.2.0...v0.3.0) (2023-02-13)


### ⚠ BREAKING CHANGES

* Using objects instead of arrays to specify pipline steps

### Features

* first dummy regressor implementation ([75b1a6e](https://github.com/mljs/ml-pipe/commit/75b1a6efef741cb7a36386ea9f6977fc92353c4d))
* implement `DummyClassifier` ([5df931a](https://github.com/mljs/ml-pipe/commit/5df931a09fa91c0c3a9d53f3c0602232cd99387b)), closes [#36](https://github.com/mljs/ml-pipe/issues/36)
* implement SVR with options, closes [#4](https://github.com/mljs/ml-pipe/issues/4) ([d4389b0](https://github.com/mljs/ml-pipe/commit/d4389b0014fd0baac1bf71f55ee4d725aeb8b3f2))
* refactor pipeline interface ([b83ca31](https://github.com/mljs/ml-pipe/commit/b83ca313107a13cc3550b82a0256935cc0d2d7e5))

## [0.2.0](https://github.com/mljs/ml-pipe/compare/v0.1.0...v0.2.0) (2022-05-17)


### Features

* compile NN upon fit call ([d15b12c](https://github.com/mljs/ml-pipe/commit/d15b12cc1d6f7007b335caa7ab8acf30ccf621fb))


### Bug Fixes

* unfitted NN problem ([c202e0e](https://github.com/mljs/ml-pipe/commit/c202e0ecda0a8d729f18b8ca6d02ae0ca172bda7))

## [0.1.0](https://github.com/mljs/ml-pipe/compare/v0.0.1...v0.1.0) (2022-05-03)


### Features

* first GDF draft ([4901214](https://github.com/mljs/ml-pipe/commit/490121425189d3c13880b12b92cad965926f0f22))
* first implementation of doFAIC ([b0d605c](https://github.com/mljs/ml-pipe/commit/b0d605c577bbc7047f513610268daea3d41ba82b))
* implement basic regression metrics ([65b74b6](https://github.com/mljs/ml-pipe/commit/65b74b6e57d426ae5d3fdf9516845f404eb15d4c))
* implemented random Gaussian matrix ([328d707](https://github.com/mljs/ml-pipe/commit/328d707e8fb304e29503bbaef1f60d7e629cb757))
* inital implementation of GDF ([a57a0b1](https://github.com/mljs/ml-pipe/commit/a57a0b1b7fafbd4d310eebc41a47ab9e0081ae26))


### Bug Fixes

* close [#17](https://github.com/mljs/ml-pipe/issues/17) ([4122d8d](https://github.com/mljs/ml-pipe/commit/4122d8d62a31fb3ad3462e479e0f40787c968353))
* random forest output shape ([22dfa19](https://github.com/mljs/ml-pipe/commit/22dfa19ae39aa1de3e6fcfdb292223c86987f0d6))

### 0.0.1 (2022-04-13)


### Features

* add FCNN ([509be57](https://github.com/mljs/ml-pipe/commit/509be57626a3d0698281f886be85ec4ccdcbfeda))
* added basic linear regressor ([7143a2e](https://github.com/mljs/ml-pipe/commit/7143a2e85b337aa063034dd6ce895a0648e798ee))
* added imbalanced test case for stratified sampling ([565a4fa](https://github.com/mljs/ml-pipe/commit/565a4fa64a28ffcde7917079f1212e5cdb239753))
* added stratification to traintestplit ([98dcda4](https://github.com/mljs/ml-pipe/commit/98dcda4429b26841f0278ce48e6e6d74d06a47b5))
* added test for RF ([0c428d0](https://github.com/mljs/ml-pipe/commit/0c428d0baa1d469a490aa7aea37a5b9b35f98a88))
* apply inverse transform, close [#5](https://github.com/mljs/ml-pipe/issues/5) ([fa88434](https://github.com/mljs/ml-pipe/commit/fa88434ef8771603aa96bb91f3983bc6604bc359))
* basic pipeline steps ([c0133b7](https://github.com/mljs/ml-pipe/commit/c0133b7ed5bd4f3f793394bacccdde2eab0be8ad))
* basic scaler implemented ([fdf9665](https://github.com/mljs/ml-pipe/commit/fdf96652548bdfead7d9cc32f4ffe0f0b06f7559))
* basic trainteastsplit implementation ([ef9225c](https://github.com/mljs/ml-pipe/commit/ef9225c2f238f5012d8d7ac05ecf1638b8c7883d))
* first idea of target transformer ([c065297](https://github.com/mljs/ml-pipe/commit/c065297c5a2755300ff2deddc7bf9e6e109db580))
* implemented MinMaxScaler ([1893058](https://github.com/mljs/ml-pipe/commit/189305832c878c849761b9897d50db917dc22e61))
* initial commit ([a59f4d9](https://github.com/mljs/ml-pipe/commit/a59f4d95f04fcf73c896c523bc30ef543d257c8f))
* really basic FCNN implementation ([5d2e1cd](https://github.com/mljs/ml-pipe/commit/5d2e1cdbd349878bd40a8f75fc99cc176d1306e7))
* simply NN options, close [#6](https://github.com/mljs/ml-pipe/issues/6) ([bd998d5](https://github.com/mljs/ml-pipe/commit/bd998d53d2bf3db435ec002e3216d4cc9fc511cb))
* start implementing models ([e9f244b](https://github.com/mljs/ml-pipe/commit/e9f244b02ad64e091fd7f33d982a9f12352de9cf))
* start implementing stratified split ([e172774](https://github.com/mljs/ml-pipe/commit/e172774070cb8473b52bc84056e762e76afe546f))
* stratified split works, but still high variance ([d184a01](https://github.com/mljs/ml-pipe/commit/d184a01fb356360a5dc4e8ca4b4f3fb4b9e42d69))
* use seedable prng ([9107682](https://github.com/mljs/ml-pipe/commit/91076822c6144ec7a49351521e17a8a9586b20de))
* use seedable prng ([c766c31](https://github.com/mljs/ml-pipe/commit/c766c3175aaa50fce36c7c72251a4b2bb53ed3c3))
* working on scaffold ([9e5416d](https://github.com/mljs/ml-pipe/commit/9e5416db8e7d1519fabbd43bbc3f5de64b90811a))


### Bug Fixes

* pipeline validation ([872b106](https://github.com/mljs/ml-pipe/commit/872b106448f3f247d247ca8659841892c978d127))


### Miscellaneous Chores

* prepare release ([9d98bf5](https://github.com/mljs/ml-pipe/commit/9d98bf594f81ea6c71bfc9d733cce88ff11513d5))
