export function clamp(num: number, min: number, max: number) {
  return num <= min ? min : num >= max ? max : num;
}

export function gaussRandom() {
  return (
    Math.sqrt(-2.0 * Math.log(Math.random())) *
    Math.cos(2.0 * Math.PI * Math.random())
  );
}
