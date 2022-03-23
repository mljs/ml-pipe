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
import { myModule } from 'ml-pipe';

const result = myModule(args);
// result is ...
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
