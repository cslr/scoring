
#include <dinrhiw/dinrhiw.h>
#include <vector>
#include <string>

using namespace whiteice;

int main(int argc, char** argv)
{
  unsigned int INPUT_DIMENSIONS = 1;
  const unsigned int DEEPNESS = 30;

  std::string examplesCSV = "examples.csv";
  std::string scoredCSV = "input.csv";
  
  // 1. load CSV example scoring (DIMENSION+1 dimensions)
  //    and to be scored data (DIMENSION dimensions)

  // 2. load and convert to complex data

  dataset<> ds, example_ds;
  dataset< math::blas_complex<float> > z_ds, z_example_ds;

  if(example_ds.importAscii(examplesCSV) == false){
    std::cout << "ERROR: Loading example data from file: '" << examplesCSV << "' FAILED."
	      << std::endl;
    return -1;
  }

  if(example_ds.getNumberOfClusters() != 1){
    std::cout << "ERROR: Loading example data from file: '" << examplesCSV << "' FAILED."
	      << std::endl;
    return -1;
  }

  if(example_ds.size(0) < 10){
    std::cout << "ERROR: Less than 10 examples in examples file: '" << examplesCSV << "'."
	      << std::endl;
    return -1;
  }

  if(example_ds.dimension(0) <= 1){
    std::cout << "ERROR: Less than 2 dimensions in examples data: '" << examplesCSV << "'."
	      << std::endl;
    return -1;
  }

  if(ds.importAscii(scoredCSV) == false){
    std::cout << "ERROR: Loading to be scored data from file: '" << scoredCSV << "' FAILED."
	      << std::endl;
    return -1;
  }

  if(ds.getNumberOfClusters() != 1){
    std::cout << "ERROR: Loading to be scored data from file: '" << scoredCSV << "' FAILED."
	      << std::endl;
    return -1;
  }

  if(ds.dimension(0) != example_ds.dimension(0)-1){
    std::cout << "ERROR: To be scored data input dimension mismatch with examples: '" << scoredCSV << "'."
	      << std::endl;
    return -1;
  }


  INPUT_DIMENSIONS = example_ds.dimension(0)-1;

  auto copy_ds = example_ds;
  copy_ds.clear();
  copy_ds.createCluster("input", INPUT_DIMENSIONS);
  copy_ds.createCluster("output", 1);

  math::vertex<> in, out;
  in.resize(INPUT_DIMENSIONS);
  out.resize(1);

  for(unsigned int i=0;i<example_ds.size(0);i++){
    auto& v = example_ds[i];
    
    for(unsigned int j=0;j<v.size()-1;j++)
      in[j] = v[j];

    out[0] = v[v.size()-1];

    copy_ds.add(0, in);
    copy_ds.add(1, out);
  }

  example_ds = copy_ds;

  ds.createCluster("output", 1);

  // convert training examples "example_ds" and to be scored data "ds"
  // to complex numbers z_example_ds, z_ds

  std::vector< math::vertex< math::blas_complex<float> > > data_in_complex;
  std::vector< math::vertex< math::blas_complex<float> > > examples_in_complex;
  std::vector< math::vertex< math::blas_complex<float> > > examples_out_complex;
  
  for(unsigned int i=0;i<ds.size(0);i++){
    math::vertex< math::blas_complex<float> > z;
    convert(z, ds[i]);
    data_in_complex.push_back(z);
  }

  for(unsigned int i=0;i<example_ds.size(0);i++){
    math::vertex< math::blas_complex<float> > z;
    convert(z, example_ds.access(0,i));
    examples_in_complex.push_back(z);

    convert(z, example_ds.access(1,i));
    examples_out_complex.push_back(z);
  }

  z_example_ds.clear();
  z_example_ds.createCluster("input", INPUT_DIMENSIONS);
  z_example_ds.createCluster("output", 1);

  z_example_ds.add(0, examples_in_complex);
  z_example_ds.add(1, examples_out_complex);

  z_example_ds.preprocess(0); // mean-variance normalization to data
  z_example_ds.preprocess(1);

  z_ds = z_example_ds;
  z_ds.clear(0);
  z_ds.clear(1);

  z_ds.add(0, data_in_complex);
  
  // 3. teach complex valued neural network
  
  nnetwork< math::blas_complex<float> >* nnet = nullptr;

  {
    std::vector<unsigned int> arch;
    arch.push_back(INPUT_DIMENSIONS);
    
    unsigned int size = 100;
    if(INPUT_DIMENSIONS > size)
      size = INPUT_DIMENSIONS;
    
    for(unsigned int i=0;i<(DEEPNESS-1);i++){
      arch.push_back(size);
    }
    
    arch.push_back(1); // scoring single value
    
    nnet = new nnetwork< math::blas_complex<float> >
      (arch,
       nnetwork< math::blas_complex<float> >::rectifier);
    
    nnet->setNonlinearity(nnet->getLayers()-1,
			  nnetwork< math::blas_complex<float> >::pureLinear);
    nnet->setResidual(true); // set residual neural network

    nnet->randomize();

  }
  
  if(nnet){
    delete nnet;
    nnet = nullptr;
  }

  return 0;
}
