// Classification des maladies du haricot à partir d'un modèle ONNX, en C++.
//
// Ce programme charge le fichier beans_resnet18.onnx exporté par le notebook
// Transfer_learning_pytorch.ipynb et classe une image passée en argument.
// Ni Python, ni PyTorch ne sont nécessaires ici : seuls ONNX Runtime et OpenCV
// le sont.
//
// Usage : ./beans_classifier <chemin_image> [chemin_modele.onnx]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <optional>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

namespace {

// ---------------------------------------------------------------------------
// Le contrat du modèle
// ---------------------------------------------------------------------------
// Le fichier .onnx ne contient que le réseau : ni le prétraitement, ni le nom
// des classes. Les constantes ci-dessous doivent donc reproduire exactement ce
// qui a été utilisé à l'entraînement, sans quoi les prédictions seront fausses
// sans le moindre message d'erreur.

constexpr int kImageSize = 224;  // T.Resize((224, 224))
constexpr int kNumClasses = 3;

// Statistiques de normalisation ImageNet : T.Normalize(mean, std)
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

// ATTENTION : cet ordre est celui d'ImageFolder, c'est-à-dire l'ordre
// alphabétique des noms de dossiers. Il fait partie du contrat du modèle :
// le réseau ne renvoie que des indices, c'est cette liste qui leur donne un sens.
const char* kClassNames[kNumClasses] = {"angular_leaf_spot", "bean_rust", "healthy"};

// Le chemin ou le modèle est sauvegardé. 
const char* kDefaultModelPath = "../pytorch-exemples/beans_resnet18.onnx";

// ---------------------------------------------------------------------------
// Prétraitement : reproduit eval_tf du notebook
// ---------------------------------------------------------------------------
// Étapes, dans cet ordre :
//   1. redimensionnement en 224x224
//   2. BGR -> RGB (OpenCV lit en BGR, le modèle attend du RGB)
//   3. conversion en float et division par 255      -> T.ToTensor()
//   4. normalisation (x - mean) / std               -> T.Normalize(mean, std)
//   5. réorganisation HWC -> CHW attendue par PyTorch/ONNX
std::vector<float> PreprocessImage(const cv::Mat& image_bgr) {
    cv::Mat resized;
    cv::resize(image_bgr, resized, cv::Size(kImageSize, kImageSize));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // 1/255 : équivalent de T.ToTensor(), qui ramène les pixels dans [0, 1]
    cv::Mat rgb_float;
    rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

    // CHW : le tenseur attendu est (1, 3, 224, 224)
    std::vector<float> tensor(static_cast<size_t>(3) * kImageSize * kImageSize);
    for (int y = 0; y < kImageSize; ++y) {
        for (int x = 0; x < kImageSize; ++x) {
            const cv::Vec3f& pixel = rgb_float.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; ++c) {
                const size_t index =
                    static_cast<size_t>(c) * kImageSize * kImageSize +
                    static_cast<size_t>(y) * kImageSize + static_cast<size_t>(x);
                tensor[index] = (pixel[c] - kMean[c]) / kStd[c]; // normalisation
            }
        }
    }
    return tensor;
}

// Softmax : convertit les logits en probabilités.
// On soustrait le maximum avant l'exponentielle pour éviter tout débordement.
std::vector<float> Softmax(const float* logits, const int count) {
    const float max_logit = *std::max_element(logits, logits + count);
    std::vector<float> probabilities(count);
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    for (float& p : probabilities) {
        p /= sum;
    }
    return probabilities;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage : " << argv[0] << " <chemin_image> [chemin_modele.onnx]\n"
                  << "Exemple : " << argv[0] << " feuille.jpg\n";
        return 1;
    }

    const std::string image_path = argv[1];
    const std::string model_path = (argc == 3) ? argv[2] : kDefaultModelPath;

    // Lecture de l'image
    const cv::Mat image_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    const cv::Mat image_bgr_copy = image_bgr.clone();
    if (image_bgr.empty()) {
        std::cerr << "Erreur : impossible de lire l'image '" << image_path << "'.\n";
        return 1;
    }
    std::cout << "Image  : " << image_path << " (" << image_bgr.cols << "x"
              << image_bgr.rows << ")\n";

    std::optional<int> predicted_id;
    std::vector<float> probabilities(3);

    try {
        // Chargement du modèle 
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "beans_classifier");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "Modèle : " << model_path << "\n";

        // On lit les noms d'entrée/sortie dans le fichier plutôt que de les
        // écrire en dur : un ré-export qui les renommerait resterait compatible.
        Ort::AllocatorWithDefaultOptions allocator;
        const Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
        const Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};

        // Temps de debut
        auto start_time = std::chrono::steady_clock::now();

        // Prétraitement de l'image
        std::vector<float> input_tensor_values = PreprocessImage(image_bgr);

        // La dimension batch du modèle est dynamique : on traite ici une seule image.
        const std::array<int64_t, 4> input_shape = {1, 3, kImageSize, kImageSize};

        const Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // Inférence 
        std::vector<Ort::Value> outputs =
            session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                        output_names, 1);

        const float* logits = outputs.front().GetTensorData<float>();

        // Résultat 
        probabilities = Softmax(logits, kNumClasses);
        
        const auto max_element_iter = std::max_element(probabilities.begin(), probabilities.end());
        predicted_id =std::optional<int>(std::distance(probabilities.begin(), max_element_iter));
        
        // Temps après inférence et obtension du résultat
        auto end_time = std::chrono::steady_clock::now();
        std::cout << "Temps d'inférence: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time -start_time).count() 
            << " [ms]" << std::endl;

        std::cout << "\nPrédiction : " << kClassNames[predicted_id.value()] << " ("
                  << std::round(probabilities[predicted_id.value()] * 100000.0f) / 1000.0f << " %)\n\n"
                  << "Détail des probabilités :\n";
        for (int i = 0; i < kNumClasses; ++i) {
            std::cout << "  " << kClassNames[i] << " : "
                      << std::round(probabilities[i] * 100000.0f) / 1000.0f << " %\n";
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "Erreur ONNX Runtime : " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << "\n";
        return 1;
    }

    if (!predicted_id.has_value()){
        std::cout << "La classification n'a pas reussi" << std::endl;
        return 1;
    }

    std::string window_name = "Prediction: " + 
                        static_cast<std::string>(kClassNames[predicted_id.value()]) + " " +
                        std::to_string(probabilities[predicted_id.value()]*100.0) + "%";

    // visualiser l'image
    cv::imshow(window_name, image_bgr_copy);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
