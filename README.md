# ml-pipe

[![NPM version][npm-image]][npm-url]
[![build status][ci-image]][ci-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

Orchestrate ML pipelines. One design different to `sklearn` pipelines is that our transformers can have the attribute `onTarget` for use in Pipelines. There, it can be convenient and to simply add a target transformation as a step into the Pipeline 

## Installation

`$ npm i ml-pipe`

## Usage

```js
import { trainTestSplit } from 'ml-pipe/modelSelection/trainTestSplit';
import {Pipeline} from 'ml-pipe/pipeline';
import {FCNN} from 'ml-pipe/estimators/neuralNetwork/fcnn'
import {StandardScaler, TargetStandardScaler} from 'ml-pipe/transformers/preprocessing/standardScaler'

const {xTrain, xTest, yTrain, yTest} = trainTestSplit(x, y, 
    {trainFraction: 0.8,  stratify: yBinned})


const pipe = new Pipeline([
    ('xScale', new StandardScaler()),
    ('yScale', new TargetStandardScaler()),
    ('model', new FCNN({inputSize: 5}))
])

await pipe.fit(XTrain, yTrain)
const predictionsTest = pipe.predict(xTest)
```


## Related package:
- [scikit.js](https://github.com/javascriptdata/scikit.js)



## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-pipe.svg
[npm-url]: https://www.npmjs.com/package/ml-pipe
[ci-image]: https://github.com/kjappelbaum/ml-pipe/workflows/Node.js%20CI/badge.svg?branch=main
[ci-url]: https://github.com/kjappelbaum/ml-pipe/actions?query=workflow%3A%22Node.js+CI%22
[codecov-image]: https://img.shields.io/codecov/c/github/kjappelbaum/ml-pipe.svg
[codecov-url]: https://codecov.io/gh/kjappelbaum/ml-pipe
[download-image]: https://img.shields.io/npm/dm/ml-pipe.svg
[download-url]: https://www.npmjs.com/package/ml-pipe
