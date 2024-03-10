use burn::backend::NdArray;
use burn::tensor::{activation::softmax, Tensor};
use image::{self, GenericImageView, Pixel};
use regex::Regex;
pub use squeezenet_burn::model::{label::LABELS, normalizer::Normalizer, squeezenet1::Model};
use std::collections::HashSet;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct InferenceResult {
    pub index: usize,
    pub probability: f32,
    pub label: String,
}

type Backend = NdArray<f32>;
const HEIGHT: usize = 224;
const WIDTH: usize = 224;

pub fn parse_image(img_data: &[u8]) -> Vec<InferenceResult> {
    let img = image::load_from_memory(img_data)
        .unwrap_or_else(|_| panic!("Failed to load image from memory"));

    let resized_img = img.resize_exact(
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let mut img_array = [[[0.0; WIDTH]; HEIGHT]; 3];

    for y in 0..224usize {
        for x in 0..224usize {
            let pixel = resized_img.get_pixel(x as u32, y as u32);
            let rgb = pixel.to_rgb();

            img_array[0][y][x] = rgb[0] as f32 / 255.0;
            img_array[1][y][x] = rgb[1] as f32 / 255.0;
            img_array[2][y][x] = rgb[2] as f32 / 255.0;
        }
    }

    let image_input = Tensor::<Backend, 3>::from_data(img_array).reshape([1, 3, HEIGHT, WIDTH]);

    let normalizer = Normalizer::new();
    let normalized_image = normalizer.normalize(image_input);
    let model = Model::<Backend>::from_embedded();
    let output = model.forward(normalized_image);
    let probablities = softmax(output, 1);
    let result = probablities.into_data().convert::<f32>().value;

    let mut vec_probablities: Vec<_> = result.iter().enumerate().collect();
    vec_probablities.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    vec_probablities.truncate(5);

    let results: Vec<InferenceResult> = vec_probablities
        .iter()
        .map(|(idx, probablity)| InferenceResult {
            index: *idx,
            probability: **probablity,
            label: LABELS[*idx].to_string(),
        })
        .collect();
    results
}

pub fn compare_results(generated: Vec<InferenceResult>, config: Vec<InferenceResult>) -> f32 {
    let mut total_similarity = 0.0;
    let re: Regex = Regex::new(r"\bn\d+\b").unwrap();

    for gen in &generated {
        for conf in &config {
            if gen.label == conf.label {
                if conf.probability <= gen.probability {
                   total_similarity += 1.0;
                } else {
                    total_similarity += gen.probability;
                }
            } else {
                let gen_label_no_commas = gen.label.replace(",", "");
                let gen_label = re.replace_all(&gen_label_no_commas, "");
                let gen_words: HashSet<_> = gen_label.split_whitespace().collect();

                let conf_label_no_commas = conf.label.replace(",", "");
                let conf_label = re.replace_all(&conf_label_no_commas, "");
                let conf_words: HashSet<_> = conf_label.split_whitespace().collect();
                let intersection = gen_words.intersection(&conf_words).count() as f32;
                let union = gen_words.union(&conf_words).count() as f32;
                let jaccard_similarity = intersection / union;

                if jaccard_similarity > 0.0 {
                    if conf.probability < gen.probability {
                        total_similarity += jaccard_similarity;
                    } else {
                        total_similarity += jaccard_similarity * gen.probability;
                    }
                }
            }
        }
    }

    // Calculate the average similarity
    total_similarity
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_image() {
        let img_data = include_bytes!("../image.jpg");
        let compare_img = include_bytes!("../image.jpeg");
        let mut results = parse_image(img_data);
        let rrs = parse_image(compare_img);
        let score = compare_results(results.clone(), rrs.clone());
        println!("Score: {}", score);
    }
}
