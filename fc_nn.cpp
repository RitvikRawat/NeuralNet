#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <cfloat>
#include <algorithm>
#include <random>
#include <cmath>

using namespace std;

vector<int> layer_size;
vector<vector<double> > z;
vector<vector<double> > a;
vector<vector<double> > b;
vector<vector<double> > error;
vector<vector<vector<double> > > w;

vector<vector<vector<double> > > train_error;
vector<vector<vector<double> > > train_activation;

vector<vector<double> > Y;
vector<vector<double> > X;

int iter_num;

void user_architect_initialise(){
	layer_size.push_back(-1);
	// int lSize;
	// cout << "Enter the number of layers (including input and output) :" << endl ;
	// cin >> lSize;
	// cout << "Enter the layer one by one:" << endl;
	// int lx;
	// for(int i = 1 ; i <= lSize ; i++){
	// 	cin >> lx;
	// 	layer_size.push_back(lx);
	// }
	layer_size.push_back(2); layer_size.push_back(4); layer_size.push_back(1);
}

void declare_b_w_a_z(){
	int L = layer_size.size()-1;
	for(int i = 0 ;i <= L ; i++){
		if(i >= 2){
			vector<double> zerolist( layer_size[i]+1 , 0);
			b.push_back(zerolist);
		}
		else{
			vector<double> zerolist;
			b.push_back(zerolist);
		}

		// if(i >= 2){
		// 	vector<double> zerolist( layer_size[i]+1 , 0);
		// 	z.push_back(zerolist);
		// }
		// else{
		// 	vector<double> zerolist;
		// 	z.push_back(zerolist);
		// }

		if(i >= 1){
			vector<double> zerolist( layer_size[i]+1 , 0);
			a.push_back(zerolist);
		}
		else{
			vector<double> zerolist;
			a.push_back(zerolist);
		}

		if(i >= 2){
			vector<vector<double> > wt( layer_size[i]+1 , vector<double> (layer_size[i-1]+1, 0) );
			w.push_back(wt);
		}
		else{
			vector<vector<double> > wt;
			w.push_back(wt);
		}
	}
	error = b;
	z = b;

}

void initialise_b_w(){
	// uniform random initialisation
	double lower_bound = -5;
	double upper_bound =  5;
	default_random_engine re;
	int L = layer_size.size()-1;
	for(int l = 2 ; l <= L ; l++){
		for(int j = 1 ; j <= layer_size[l] ; j++){
			for(int k = 1 ; k <= layer_size[l-1] ; k++){
				uniform_real_distribution<double> unif(lower_bound,upper_bound);
   				double a_random_double = unif(re);
				// cout << a_random_double << endl;
				w[l][j][k] = a_random_double;
			}
		}
	}
}

void print_weights(){
	cout << "Network weights -------------------------- " << endl;
	int L = layer_size.size()-1;
	for(int l = 2 ; l <= L ; l++){
		for(int j = 1 ; j <= layer_size[l] ; j++){
			for(int k = 1 ; k <= layer_size[l-1] ; k++){
				cout << "Layer " << l-1 << " neuron "<< k <<" to Layer " << l << " neuron " << j << " is "<< w[l][j][k] << endl;
			}
		}
	}
}

void print_biases(){
	cout << "Network biases -------------------------- " << endl;
	int L = layer_size.size()-1;
	for(int l = 2 ; l <= L ; l++){
		for(int j = 1; j <= layer_size[l] ; j++){
			cout << "Layer " << l << " neuron " << j << " is " << b[l][j] << endl;
		}
	}
}

vector<double> hadamard(vector<double> x1, vector<double> x2){
	vector<double> ret (x1.size() , 0);
	for(int i = 1 ; i <= x1.size()-1 ; i++){
		ret[i] = x1[i] * x2[i];
	}
	return ret;
}

void test_hadamard(){
	vector<double> a(6, 0) , b(6,0);
	for(int i = 1 ; i <= 5 ; i++){
		a[i] = (double)i;
		b[i] = (double)(i + 2.5);
	}
	vector<double> c = hadamard(a,b);
	for(int i = 0 ; i < c.size() ; i++){
		cout << c[i] << endl;
	}
}

vector<double> trans_mat_mul_vect(vector<vector<double> > &M, vector<double> &v){
	vector<vector<double> > A( M[0].size() , vector<double> (M.size(), 0) );
	for(int i = 0 ; i < M.size() ; i++){
		for(int j = 0 ; j < M[0].size() ; j++){
			A[j][i] = M[i][j];
		}
	}
	int m = A.size()-1, n = A[0].size()-1;
	vector<double> ret(m+1, 0);
	for(int i = 1 ; i <= m ; i++){
		double val = 0;
		for(int j = 1 ; j <= n ; j++){
			val += A[i][j] * v[j];
		}
		ret[i] = val;
	}
	return ret;
}

vector<double> mat_mul_vect(vector<vector<double> > &A, vector<double> &v){
	int m = A.size()-1, n = A[0].size()-1;
	vector<double> ret(m+1, 0);
	for(int i = 1 ; i <= m ; i++){
		double val = 0;
		for(int j = 1 ; j <= n ; j++){
			val += A[i][j] * v[j];
		}
		ret[i] = val;
	}
	return ret;
}

void test_trans_mat_mul_vect(){
	vector<double> a(3, 0) , r1(2,0), r2(2,0), r3(2,0);
	for(int i = 1 ; i <= 2 ; i++){
		a[i] = (double)i;
	}
	for(int i = 0 ; i <= 1 ; i++){
		r1[i] = (double)i+1.5;
		r2[i] = (double)i+2.5;
		r3[i] = (double)i+3.5;
	}
	vector<vector<double> > M;
	M.push_back(r1); M.push_back(r2); M.push_back(r3);

	vector<double> res = trans_mat_mul_vect(M,a);
	for(int i = 0 ; i < res.size() ; i++){
		cout << res[i] << endl;
	}
}

void test_mat_mul_vect(){
	vector<double> a(3, 0) , r1(3,0), r2(3,0);
	for(int i = 0 ; i <= 2 ; i++){
		a[i] = (double)i+1;
	}
	for(int i = 0 ; i <= 2 ; i++){
		r1[i] = (double)i+1;
		r2[i] = (double)i+2;
	}
	vector<vector<double> > M;
	M.push_back(r1); M.push_back(r2);

	vector<double> res = mat_mul_vect(M,a);
	for(int i = 0 ; i < res.size() ; i++){
		cout << res[i] << endl;
	}
}

vector<double> add_vec(vector<double> v1, vector<double> v2){
	vector<double> ret (v1.size() , 0);
	for(int i = 1 ; i < v1.size() ; i++){
		ret[i] = v1[i] + v2[i];
	}
	return ret;
}

vector<double> sub_vec(vector<double> v1, vector<double> v2){
	vector<double> ret (v1.size() , 0);
	for(int i = 1 ; i < v1.size() ; i++){
		ret[i] = v1[i] - v2[i];
	}
	return ret;
}

void test_add_vec(){
	vector<double> tt1(3,0), tt2(3,0);
	for(int i = 0 ; i <= 2 ; i++){
		tt1[i] = (double)i+1;
		tt2[i] = (double)i+2;
	}
	vector<double> sol = add_vec(tt1,tt2);
	for(int i = 0 ; i < sol.size() ; i++){
		cout << sol[i] << endl;
	}
}

void data_initialiser(){
	vector<double> tt1(3,0), tt2(3,0), tt3(3,0), tt4(3,0);
	tt1[1] = 1.0; tt1[2] = 0.0;
	tt2[1] = 0.0; tt2[2] = 1.0;
	tt3[1] = 1.0; tt3[2] = 0.0;
	tt4[1] = 1.0; tt4[2] = 1.0;
	X.push_back(tt1); X.push_back(tt2); X.push_back(tt3); X.push_back(tt4);

	vector<double> y;
	y.push_back(0); y.push_back(1.0); Y.push_back(y); y.clear();
	y.push_back(0); y.push_back(1.0); Y.push_back(y); y.clear();
	y.push_back(0); y.push_back(1.0); Y.push_back(y); y.clear();
	y.push_back(0); y.push_back(0); Y.push_back(y); y.clear();

}

vector<double> sigmoid(vector<double> v){
	vector<double> ret(v.size() , 0);
	for(int i = 1 ; i < v.size() ; i++){
		// ret[i] = pow(2.71828, v[i]) / ( pow(2.71828, v[i]) + 1) ;
		ret[i] = 1 / ( pow(2.71828, -1.0 *v[i]) + 1) ;
	}
	return ret;
}

vector<double> sigmoid_prime(vector<double> v){
	vector<double> ret(v.size() , 0);
	for(int i = 1 ; i < v.size() ; i++){
		// ret[i] = pow(2.71828, v[i]) / ( pow(2.71828, v[i]) + 1) ;
		double t = 1 / ( pow(2.71828, -1.0 *v[i]) + 1) ;
		ret[i] = t * (1-t);
	}
	return ret;
}

void test_sigmoid(){
	vector<double> v;
	v.push_back(0); v.push_back(0); v.push_back(-9); v.push_back(9); v.push_back(-100);v.push_back(100); 
	vector<double> sol = sigmoid(v);
	for(int i = 0 ; i < sol.size() ; i++){
		cout << sol[i] << endl;
	}
}

void print_outputs(){
	cout << "Output of the neurons ----------------------------------" << endl;
	int L = layer_size.size()-1;
	for(int l = 2 ; l <= L ; l++){
		cout << "Layer " << l << " outputs :" << endl;
		for(int j = 1 ; j <= layer_size[l] ; j++){
			cout << z[l][j] << endl;
		}
	}
}

void print_activations(){
	cout << "Activations of the neurons ----------------------------------" << endl;
	int L = layer_size.size()-1;
	for(int l = 1 ; l <= L ; l++){
		cout << "Layer " << l << " activations :" << endl;
		for(int j = 1 ; j <= layer_size[l] ; j++){
			cout << a[l][j] << endl;
		}
	}
}

vector<vector<double> > outer_prod(vector<double> v1, vector<double> v2){
	vector<vector<double> > ret(v1.size(), vector<double> (v2.size(), 0) );
	int m = v1.size()-1, n = v2.size()-1;
	for(int i = 1 ; i <= m ; i++){
		for(int j = 1 ; j <= n ; j++){
			ret[i][j] = v1[i]*v2[j];
		}
	}
	return ret;
}

void test_outer_prod(){
	vector<double> v1(4,0), v2(3,0);
	v1[1] = 1; v1[2] = 2; v1[3] = 3;
	v2[1] = 10; v2[2] = 20;
	vector<vector<double> > sol = outer_prod(v1,v2);
	for(int i = 1 ; i < sol.size() ; i++){
		for(int j = 1 ; j < sol[0].size() ; j++){
			cout << sol[i][j] << " " ;
		}
		cout << endl;
	}
}

vector<vector<double> > add_mat(vector<vector<double> > M1, vector<vector<double> > M2){
	vector<vector<double> > ret(M1.size() , vector<double> (M2.size(), 0) );
	for(int i = 0 ; i < ret.size() ; i++){
		for(int j = 0 ; j < ret[0].size() ; j++){
			ret[i][j] = M1[i][j] + M2[i][j];
		}
	}
	return ret;
}

void matrix_scale(float lambda, vector<vector<double> > &M){
	for(int i = 0 ; i < M.size() ; i++){
		for(int j = 0 ; j < M[0].size() ; j++){
			M[i][j] = lambda * M[i][j];
		}
	}
}

void vector_scale(float lambda, vector<double> &v){
	for(int i = 0 ; i < v.size() ; i++){
		v[i] = lambda * v[i];
	}
}

double L2_norm_sq(vector<double> v){
	double ret = 0;
	for(int i = 0 ; i < v.size() ; i++){
		ret += (v[i]*v[i]);
	}
	return ret;
}

void forward_backward(){
	int L = layer_size.size()-1;
	train_error.clear();
	train_activation.clear();
	double cost;
	for(int i = 0 ; i < X.size() ; i++){
		// set activation for 1st layer as input
		a[1] = X[i];
		// forward pass
		for(int l = 2 ;l <= L ; l++){
			z[l] = add_vec( mat_mul_vect(w[l], a[l-1]) , b[l] );
			a[l] = sigmoid(z[l]);
		}
		train_activation.push_back(a);
		// Output error error[L]
		if(iter_num % 1000 == 0){
			vector<double> cost_vector = sub_vec(a[L], Y[i]);
			cost = L2_norm_sq(cost_vector);
			// cout << a[L][1]-Y[i][1] << endl;
		}
		error[L] = hadamard( sub_vec(a[L],Y[i]) , sigmoid_prime(z[L]) );
		for(int l = L-1 ; l >= 2 ; l--){
			error[l] =  hadamard( trans_mat_mul_vect(w[l+1], error[l+1]) , sigmoid_prime(z[l]) );
		}
		train_error.push_back(error);
	}
	if(iter_num % 1000 == 0){
		cout << "Iteration number "<< iter_num <<" : "<< cost << endl;
	}
}

void parameter_update(double learning_rate){
	int L = layer_size.size()-1;
	for(int l = L ; l >= 2 ; l--){
		// update weights
		vector<vector<double> >  w_change(w[l].size() , vector<double> ( w[l][0].size(), 0));
		for(int i = 0 ; i < X.size() ; i++){
			vector<vector<double> > o_p = outer_prod(train_error[i][l], train_activation[i][l-1]);
			w_change = add_mat(w_change, o_p);
		}
		matrix_scale( (learning_rate/(double)X.size()) , w_change);
		for(int j = 1 ; j <= layer_size[l] ; j++){
			w[l][j] = sub_vec( w[l][j], w_change[j]);
		}
		// update biases
		vector<double> b_change(b[l].size(), 0);
		for(int i = 0 ; i < X.size() ; i++){
			b_change = add_vec(b_change, train_error[i][l]);
		}
		vector_scale( (learning_rate/(double)X.size()) , b_change);
		b[l] = sub_vec(b[l] , b_change);
	}
}

void train_NN(int iter){
	iter_num = 0;
	for(int i = 1 ; i <= iter ; i++){
		forward_backward();
		parameter_update(0.1);
		iter_num++;
	}
}

void predict_NN(){
	int L = layer_size.size()-1;
	vector<vector<vector<double> > > results;
	for(int i = 0 ; i < X.size() ; i++){
		// set activation for 1st layer as input
		a[1] = X[i];
		// forward pass
		for(int l = 2 ;l <= L ; l++){
			z[l] = add_vec( mat_mul_vect(w[l], a[l-1]) , b[l] );
			a[l] = sigmoid(z[l]);
		}
		results.push_back(a);
	}
	cout << "Prediction by NN -------------------- " << endl;
	for(int i = 0 ; i < results.size() ; i++){
		cout << "Result " << i << " is "<< endl;
		for(int j = 1 ; j <= layer_size[L] ; j++){
			cout << results[i][L][j] << " ";
		}
		cout << endl;
	}
	cout << "-------------------------------------" << endl;
}

int main(){
	user_architect_initialise();
	declare_b_w_a_z();
	initialise_b_w();
	// test_hadamard(); test_trans_mat_mul_vect();
	data_initialiser();
	// test_sigmoid();
	// test_outer_prod();
	train_NN(100000);
	predict_NN();
	// print_biases();
	// print_weights();
	// print_outputs();
	// print_activations();

	// for(int i = 0 ; i < z.size() ; i++){
	// 	cout << z[i].size() << endl;
	// }
	return 0;
}