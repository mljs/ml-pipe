import { argsort } from '../array';

describe('test argsort', () => {
  it('test argsort - default ascending', () => {
    const array = [1, 3, 2, 4];
    const sorted = argsort(array);
    expect(sorted).toStrictEqual([0, 2, 1, 3]);
  });

  it('test argsort - descending', () => {
    const array = [1, 3, 2, 4];
    const sorted = argsort(array, false);
    expect(sorted).toStrictEqual([3, 1, 2, 0]);
  });
});
