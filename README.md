# Image Similarity Comparison in Rust

This Rust library provides functionality to parse images and compare their inference results. The primary focus of this library is the `compare_results` function, which calculates a similarity score between two sets of inference results.

## Getting Started

To use this library, include it in your Rust project by adding the following to your `Cargo.toml`:

```
[dependencies]
image-similarity = { version = "0.1.0", path = "<path-to-library>" }
```

## Usage

The main functionality of this library is provided by two functions: `parse_image` and `compare_results`.

### `parse_image`

The `parse_image` function takes a byte slice representing an image and returns a vector of `InferenceResult` objects. Each `InferenceResult` contains an index, a probability, and a label.

```
let img_data = include_bytes!("../image.jpg");
let results = parse_image(img_data);
```

### `compare_results`

The `compare_results` function is the heart of this library. It takes two vectors of `InferenceResult` objects and calculates a similarity score between them.

The similarity score is calculated as follows:

1. For each pair of `InferenceResult` objects (one from each vector), if the labels are exactly the same, the similarity score is increased by 1.0 or by the probability of the generated result, whichever is smaller.

2. If the labels are not exactly the same, but there are some words in common between them, the Jaccard similarity of the labels is calculated. This is the size of the intersection of the sets of words in the labels divided by the size of the union of the sets of words. If the Jaccard similarity is greater than 0.0, it is added to the similarity score, multiplied by the probability of the generated result if it is smaller than the probability of the config result.

3. Score above 1 is considered as matching.... [0-5]
The function returns the total similarity score.

```
let img_data1 = include_bytes!("../image1.jpg");
let img_data2 = include_bytes!("../image2.jpg");
let results1 = parse_image(img_data1);
let results2 = parse_image(img_data2);
let score = compare_results(results1, results2);
println!("Score: {}", score);
```

## Testing

The library includes a test module with a test for the `parse_image` function. This test parses two images and compares their results.

```
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_image() {
        let img_data = include_bytes!("../image.jpg");
        let compare_img = include_bytes!("../image.jpeg");
        let results = parse_image(img_data);
        let compare_results = parse_image(compare_img);
        let score = compare_results(results, compare_results);
        println!("Score: {}", score);
    }
}
```

## Contributing

Contributions are welcome. Please submit a pull request with any improvements or bug fixes.

## License

This project is licensed under the MIT License.