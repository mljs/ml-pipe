import { StandardScaler } from '../../transformers/preprocessing/standardScaler';
import { trainingSet, labels, correct } from '../../utils/testHelpers';
import { Pipeline } from '../pipeline';

describe('test pipeline', () => {
  it('invalid pipeline', () => {
    expect(() => {
      new Pipeline([['bla', new StandardScaler()]]);
    }).toThrow('Last step should be an estimator but is not.');
  });
});
