#include <map>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;


int main(int argc, char *argv[])
{
	LOG(INFO) << argv[0] << "path_net_prototxt path_caffemodel";

	Caffe::set_mode(Caffe::GPU);
	int device_id = 0;
	Caffe::SetDevice(device_id);

	// Load net
    const std::string path_net_prototxt(argv[1]);
	Net<float> net(path_net_prototxt, TEST);

	// Load cafemodel
	const std::string path_caffemodel(argv[2]);
	net.CopyTrainedLayersFrom(path_caffemodel);

	int isample = 0;
	int sample_id = 0;
	std::map<int, int> voting;

	int num_models = 0;
	int num_correct_votings = 0;

	// iterate over batches in data
	for (int iters = 0; iters < 925; iters++) {
		float loss = 0.0;
		vector<Blob<float>*> results = net.ForwardPrefilled(&loss);
		const float* predicted = results[2]->cpu_data();
		//LOG(INFO)<< "results: " << results.size();

		//LOG(INFO)<< "------------- " << " batch: " << iters;

		// Get probabilities
		const boost::shared_ptr<Blob<float> >& prob = net.blob_by_name("prob");
		const float* probs_out = prob->cpu_data();
		
		// Get argmax results
		const boost::shared_ptr<Blob<float> >& label = net.blob_by_name("label");
		const float* labels_out = label->cpu_data();
		
		// iterate over sample in batch
		for (int i = 0; i < 32; i++) 
		{
			LOG(INFO) << "sample: " << sample_id << ", label: " << labels_out[i*label->height() + 0] << ", pred: " << predicted[i];
			voting[(int)predicted[i]]++;
			sample_id++;

			isample++;
			if (isample == 12) 
			{
				int class_id = 0;
				int class_votes = 0;
				for(std::map<int,int>::iterator it = voting.begin(); it != voting.end(); ++it) {
					if (class_votes < it->second) {
						class_id = it->first;
						class_votes = it->second;
					}
				}
				voting.clear();
				isample = 0;
				//LOG(INFO) << "label: " << labels_out[i*label->height() + 0] << ", voting:" << class_id;

				num_models++;
				if ( labels_out[i*label->height() + 0] == class_id) { num_correct_votings++; }
			}
		}
		//LOG(INFO)<< "-------------";
	}

	//LOG(INFO)<< "num models: " << num_models << ", num correct votes: " << num_correct_votings;
	//LOG(INFO)<< "accuracy with voting: " << ((float)num_correct_votings / (float)num_models);


	return 0;
}