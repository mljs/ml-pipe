import { Matrix } from 'ml-matrix';

import { MinMaxScaler } from '../MinMaxScaler';

describe('test MinMaxScaler', () => {
  it('two cols', () => {
    const data = new Matrix([
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1],
    ]);

    const scaler = new MinMaxScaler();
    scaler.fit(data);
    const expected = new Matrix([
      [0, 0],
      [0, 0],
      [1, 1],
      [1, 1],
    ]);

    const transformed = scaler.transform(data);
    expect(transformed).toStrictEqual(expected);
    expect(scaler.transform(new Matrix([[2, 2]]))).toStrictEqual(
      new Matrix([[2, 2]]),
    );
    expect(scaler.inverseTransform(transformed)).toStrictEqual(data);
  });

  it('constant column', () => {
    const data = new Matrix([[1, 1, 1, 1, 1, 1, 1, 1]]);

    const scaler = new MinMaxScaler();
    scaler.fit(data);
    const expected = new Matrix([[0, 0, 0, 0, 0, 0, 0, 0]]);

    const transformed = scaler.transform(data);
    expect(transformed).toStrictEqual(expected);

    expect(scaler.inverseTransform(transformed)).toStrictEqual(data);
  });
});
