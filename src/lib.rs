use squeezenet_burn::model::{label::LABELS, normalizer::Normalizer, squeezenet1::Model};
use burn::backend::NdArray;
use burn::tensor::{Tensor, activation::softmax};
use image::{self, GenericImageView, Pixel};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    index: usize,
    probability: f32,
    label: String,
}


type Backend = NdArray<f32>;
const HEIGHT: usize = 224;
const WIDTH: usize = 224;

pub fn parse_image(img_data: &[u8]) -> Vec<InferenceResult> {
    let img = image::load_from_memory(img_data).unwrap_or_else(|_| panic!("Failed to load image from memory"));

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



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_image() {
        let img_data = include_bytes!("../test.png");
        let results = parse_image(img_data);
        for result in results {
            println!("Index: {}, Probability: {}, Label: {}", result.index, result.probability, result.label);
        }
    }
}
