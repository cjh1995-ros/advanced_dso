#include "CommonInclude.hpp"


class SmallOptimizer
{
private:
    Vector8f jac_;
    Vector8f b_;
    Matrix88f hess_;
    


    cv::Mat ref_img_;
    std::vector<float> photometric_params_;
    std::vector<cv::Mat> grad_img_;
    std::vector<Eigen::Vector3f> target_point_;
    Eigen::Vector<float, 4> camera_params_;
    std::vector<float> residual_;

public:
    SmallOptimizer(const cv::Mat ref_img,
                   const std::vector<cv::Mat> grad_img, 
                   const std::vector<float> photometric_params,
                   const std::vector<Eigen::Vector3f> target_point, 
                   const Eigen::Vector<float, 4> camera_params,
                   const std::vector<float> residual)
        : camera_params_(camera_params)
    {
        // Copy ref_img to ref_img_
        ref_img.copyTo(ref_img_);

        // Copy photometric_params to photometric_params_
        for (int i=0; i<photometric_params.size(); i++)
            photometric_params_.push_back(photometric_params[i]);

        // Copy grad_img to grad_img_        
        for (int i=0; i<grad_img.size(); i++)
            grad_img_.push_back(grad_img[i]);

        // Copy target_point to target_point_
        for (int i=0; i<target_point.size(); i++)
            target_point_.push_back(target_point[i]);

        // Copy residual to residual_
        for (int i=0; i<residual.size(); i++)
            residual_.push_back(residual[i]);

        jac_.setZero();
        hess_.setZero();
    };

    /// @brief only eigen. it takes 65sec.
    inline void calcJacobian();
    inline void calcHessian(){};
};


void SmallOptimizer::calcJacobian()
{
    float fx = camera_params_(0);
    float fy = camera_params_(1);
    float cx = camera_params_(2);
    float cy = camera_params_(3);
    float a = photometric_params_[0];
    float b = photometric_params_[1];
    int n_points = target_point_.size();

    Eigen::VectorXf jac_rho(residual_.size());
    Eigen::VectorXf jac_y(8);

    Eigen::MatrixXf hess_y_y(8, 8);
    Eigen::MatrixXf hess_rho_y(residual_.size(), 8);
    Eigen::MatrixXf hess_y_rho(8, residual_.size());
    Eigen::MatrixXf hess_rho_rho(residual_.size(), residual_.size());

    Eigen::VectorXf b_y(8);
    Eigen::VectorXf b_rho(residual_.size());

    for (int i=0; i<n_points; i++)
    {
        // select random pixels and gradient values in image
        float u = target_point_[i](0);
        float v = target_point_[i](1);
        float rho = target_point_[i](2);
        float grad_x = grad_img_[0].at<float>((int)v, (int)u);
        float grad_y = grad_img_[1].at<float>((int)v, (int)u);
        float img_val = ref_img_.at<float>((int)v, (int)u);

        jac_y[0] += grad_x * rho * fx;
        jac_y[1] += grad_y * rho * fy;
        jac_y[2] += -rho * (grad_x * fx * u + grad_y * fy * u);
        jac_y[3] += -grad_x * u * v * fx - grad_y * fy * (1 + v * v);
        jac_y[4] += grad_x * (1 + u * u) + grad_y * u * v;
        jac_y[5] += -grad_x * fx * v + grad_y * fy * u;
        jac_y[6] += std::exp(a) * img_val; // original -> image value at image 1
        jac_y[7] += -1;

        jac_rho[i] = (grad_x * fx + grad_y * fy) * rho;

        hess_y_y += jac_ * jac_.transpose();
        hess_rho_y.block<1, 8>(i, 0) = jac_rho[i] * jac_.transpose();
        hess_y_rho.block<8, 1>(0, i) = jac_ * jac_rho[i];
        hess_rho_rho(i, i) = jac_rho[i] * jac_rho[i];
        
        b_y += -jac_y * residual_[i];
        b_rho[i] = -jac_rho[i] * residual_[i];
    }

    // check time with using chrono for 1 itr
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXf hess_sc = hess_y_rho * hess_rho_rho.inverse() * hess_rho_y;
    Eigen::VectorXf b_sc = hess_y_rho * hess_rho_rho.inverse() * b_rho;

    // solve for x_y
    Eigen::VectorXf x_y = (hess_y_y - hess_sc).inverse() * (b_y - b_sc);

    // solve for x_rho
    Eigen::VectorXf x_rho = hess_rho_rho.inverse() * (b_rho - hess_rho_y * x_y);

    // end timer with seconds
    auto end = std::chrono::high_resolution_clock::now();

    auto gap = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Time taken by function: " << gap << " microseconds" << std::endl;
}


int main()
{
    // gen normal camera params
    Eigen::Vector<float, 4> camera_params;
    camera_params << 300, 300, 320, 240;

    // gen target image with random values
    cv::Mat img = cv::Mat::zeros(640, 480, CV_32FC1);
    cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(1));

    // gen target image gradient with random values
    std::vector<cv::Mat> grad_imgs;

    cv::Mat grad_x = cv::Mat::zeros(640, 480, CV_32FC1);
    cv::Mat grad_y = cv::Mat::zeros(640, 480, CV_32FC1);
    cv::randu(grad_x, cv::Scalar::all(0), cv::Scalar::all(1));
    cv::randu(grad_y, cv::Scalar::all(0), cv::Scalar::all(1));

    grad_imgs.push_back(grad_x);
    grad_imgs.push_back(grad_y);

    // gen target 3d points with random values
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0,1.0);
    std::uniform_int_distribution<int> distributionX(0,639);
    std::uniform_int_distribution<int> distributionY(0,479);

    std::vector<Eigen::Vector3f> vec;

    for(int i = 0; i < 1500; ++i) {
        Eigen::Vector3f v;
        v << distributionX(generator), distributionY(generator), std::abs(distribution(generator));
        vec.push_back(v);
    }

    // gen residual with random values
    std::vector<float> residual;
    for(int i = 0; i < 1500; ++i) {
        residual.push_back(distribution(generator));
    }

    std::vector<float> photometric_params;
    photometric_params.push_back(0.1);
    photometric_params.push_back(0.2);

    // print all values information
    std::cout << "camera_params: " << camera_params.transpose() << std::endl;
    std::cout << "img information: (width, height)" << img.size() << std::endl;
    std::cout << "grad_imgs information: (width, height)" << grad_imgs[0].size() << std::endl;
    std::cout << "number of random pixels: " << vec.size() << std::endl;
    std::cout << "number of random residual: " << residual.size() << std::endl;

    SmallOptimizer small_optimizer(img, grad_imgs, photometric_params, vec, camera_params, residual);
    small_optimizer.calcJacobian();

    return 0;
}