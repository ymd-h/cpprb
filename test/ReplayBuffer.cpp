#include <iostream>
#include <chrono>
#include <tuple>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <future>
#include <thread>
#include <numeric>

#include <ReplayBuffer.hh>

#include "unittest.hh"

using namespace std::literals;

using Observation = double;
using Action = double;
using Reward = double;
using Done = double;
using Priority = double;

const auto cores = std::thread::hardware_concurrency();
using cores_t = std::remove_const_t<decltype(cores)>;

void test_DimensionalBuffer(){
  constexpr const auto obs_dim = 3ul;

  constexpr const auto N_buffer_size = 1024ul;
  constexpr const auto N_times = 1000ul;

  auto dm = ymd::DimensionalBuffer<Observation>{N_buffer_size,obs_dim};
  auto v = std::vector<Observation>{};
  std::generate_n(std::back_inserter(v),obs_dim,
		  [i=0ul]()mutable{ return Observation(i++); });

  std::cout << "DimensionalBuffer: " << std::endl;
  Observation* obs_ptr = nullptr;
  dm.get_data(0ul,obs_ptr);
  std::cout << " DimensionalBuffer.data(): " << obs_ptr<< std::endl;
  std::cout << "*DimensionalBuffer.data(): " << *obs_ptr << std::endl;

  dm.store_data(v.data(),0ul,0ul,1ul);
  ALMOST_EQUAL(obs_ptr[0],0ul);
  ALMOST_EQUAL(obs_ptr[1],1ul);
  ALMOST_EQUAL(obs_ptr[2],2ul);

  for(auto n = 0ul; n < N_times; ++n){
    auto next_index = std::min(n*obs_dim % N_buffer_size,N_buffer_size-1);
    dm.store_data(v.data(),0ul,next_index,1ul);
    ALMOST_EQUAL(obs_ptr[next_index * obs_dim + 0],0ul);
    ALMOST_EQUAL(obs_ptr[next_index * obs_dim + 1],1ul);
    ALMOST_EQUAL(obs_ptr[next_index * obs_dim + 2],2ul);
  }
}

void test_PrioritizedSampler(){
  constexpr const auto N_buffer_size = 1024ul;
  constexpr const auto N_batch_size = 16ul;
  constexpr const auto N_step = 3 * N_buffer_size;

  constexpr const auto alpha = 0.7;
  constexpr const auto beta = 0.4;

  constexpr const auto LARGE_P = 1e+10;

  std::cout << std::endl;
  std::cout << "PrioritizedSampler" << std::endl;
  auto ps = ymd::CppPrioritizedSampler(N_buffer_size,alpha);
  for(auto i = 0ul; i < N_step; ++i){
    ps.set_priorities(i % N_buffer_size,0.5);
  }

  auto ps_w = std::vector<Priority>{};
  auto ps_i = std::vector<std::size_t>{};

  ps.sample(N_batch_size,beta,ps_w,ps_i,N_buffer_size);

  ymd::show_vector(ps_w,"weights [0.5,...,0.5]");
  ymd::show_vector(ps_i,"indexes [0.5,...,0.5]");

  for(auto& w : ps_w){
    ALMOST_EQUAL(w,1.0);
  }

  ps_w[0] = LARGE_P;
  ps.update_priorities(ps_i,ps_w);
  ps.sample(N_batch_size,beta,ps_w,ps_i,N_buffer_size);
  ymd::show_vector(ps_w,"weights [0.5,.,1e+10,..,0.5]");
  ymd::show_vector(ps_i,"indexes [0.5,.,1e+10,..,0.5]");

  ALMOST_EQUAL(ps.get_max_priority(),LARGE_P);

  ALMOST_EQUAL(std::accumulate(ps_w.begin(),ps_w.end(),0.0) / ps_w.size(),LARGE_P);
}

void test_SelectiveEnvironment(){
  constexpr const auto obs_dim = 3ul;
  constexpr const auto act_dim = 1ul;
  constexpr const auto episode_len = 4ul;
  constexpr const auto Nepisodes = 10ul;

  auto se = ymd::CppSelectiveEnvironment<Observation,Action,Reward,Done>(episode_len,
									 Nepisodes,
									 obs_dim,
									 act_dim);

  std::cout << std::endl
	    << "SelectiveEnvironment("
	    << "episode_len=" << episode_len
	    << ",Nepisodes=" << Nepisodes
	    << ",obs_dim=" << obs_dim
	    << ",act_dim=" << act_dim
	    << ")" << std::endl;

  EQUAL(se.get_next_index(),0ul);
  EQUAL(se.get_stored_size(),0ul);
  EQUAL(se.get_stored_episode_size(),0ul);
  EQUAL(se.get_buffer_size(),episode_len*Nepisodes);

  auto obs = std::vector(obs_dim*(episode_len+1),Observation{1});
  auto act = std::vector(act_dim*episode_len,Action{1.5});
  auto rew = std::vector(episode_len,Reward{1});
  auto done = std::vector(episode_len,Done{0});
  done.back() = Done{1};

  // Add 1-step
  se.store(obs.data(),act.data(),rew.data(),obs.data()+1,done.data(),1ul);
  auto [obs_,act_,rew_,next_obs_,done_,ep_len] = se.get_episode(0);
  ymd::show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  ymd::show_pointer(act_,se.get_stored_size()*act_dim,"act");
  ymd::show_pointer(rew_,se.get_stored_size(),"rew");
  ymd::show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  ymd::show_pointer(done_,se.get_stored_size(),"done");

  EQUAL(ep_len,1ul);
  EQUAL(se.get_next_index(),1ul);
  EQUAL(se.get_stored_size(),1ul);
  EQUAL(se.get_stored_episode_size(),1ul);

  // Add remained 3-steps
  se.store(obs.data()+1,act.data()+1,rew.data()+1,obs.data()+2,done.data()+1,
	   episode_len - 1ul);
  se.get_episode(0,ep_len,obs_,act_,rew_,next_obs_,done_);
  ymd::show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  ymd::show_pointer(act_,se.get_stored_size()*act_dim,"act");
  ymd::show_pointer(rew_,se.get_stored_size(),"rew");
  ymd::show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  ymd::show_pointer(done_,se.get_stored_size(),"done");

  EQUAL(ep_len,episode_len);
  EQUAL(se.get_next_index(),episode_len);
  EQUAL(se.get_stored_size(),episode_len);
  EQUAL(se.get_stored_episode_size(),1ul);

  // Try to get non stored episode
  se.get_episode(1,ep_len,obs_,act_,rew_,next_obs_,done_);
  EQUAL(ep_len,0ul);

  // Add shorter epsode
  se.store(obs.data()+1,act.data()+1,rew.data()+1,obs.data()+2,done.data()+1,
	   episode_len - 1ul);
  se.get_episode(0,ep_len,obs_,act_,rew_,next_obs_,done_);
  ymd::show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  ymd::show_pointer(act_,se.get_stored_size()*act_dim,"act");
  ymd::show_pointer(rew_,se.get_stored_size(),"rew");
  ymd::show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  ymd::show_pointer(done_,se.get_stored_size(),"done");

  EQUAL(se.get_next_index(),2*episode_len - 1ul);
  EQUAL(se.get_stored_size(),2*episode_len - 1ul);
  EQUAL(se.get_stored_episode_size(),2ul);

  se.get_episode(1,ep_len,obs_,act_,rew_,next_obs_,done_);
  EQUAL(ep_len,episode_len - 1ul);

  // Delete non existing episode
  EQUAL(se.delete_episode(99),0ul);
  EQUAL(se.get_next_index(),2*episode_len - 1ul);
  EQUAL(se.get_stored_size(),2*episode_len - 1ul);
  EQUAL(se.get_stored_episode_size(),2ul);

  // Delete 0
  se.delete_episode(0);
  se.get_episode(0,ep_len,obs_,act_,rew_,next_obs_,done_);
  ymd::show_pointer(obs_,se.get_stored_size()*obs_dim,"obs");
  ymd::show_pointer(act_,se.get_stored_size()*act_dim,"act");
  ymd::show_pointer(rew_,se.get_stored_size(),"rew");
  ymd::show_pointer(next_obs_,se.get_stored_size()*obs_dim,"next_obs");
  ymd::show_pointer(done_,se.get_stored_size(),"done");
  EQUAL(se.get_next_index(),episode_len - 1ul);
  EQUAL(se.get_stored_size(),episode_len - 1ul);
  EQUAL(se.get_stored_episode_size(),1ul);

  // Add shorter epsode with not terminating
  se.store(obs.data(),act.data(),rew.data(),obs.data()+1,done.data(),
	   episode_len - 1ul);
  EQUAL(se.get_next_index(),2*episode_len - 2ul);
  EQUAL(se.get_stored_size(),2*episode_len - 2ul);
  EQUAL(se.get_stored_episode_size(),2ul);

  // Delete half-open episode
  se.delete_episode(1);
  EQUAL(se.get_next_index(),episode_len - 1ul);
  EQUAL(se.get_stored_size(),episode_len - 1ul);
  EQUAL(se.get_stored_episode_size(),1ul);

  // Add shorter epsode with not terminating
  se.store(obs.data(),act.data(),rew.data(),obs.data()+1,done.data(),
	   episode_len - 1ul);
  EQUAL(se.get_next_index(),2*episode_len - 2ul);
  EQUAL(se.get_stored_size(),2*episode_len - 2ul);
  EQUAL(se.get_stored_episode_size(),2ul);

  // Delete 0 when finishing half-open episode
  se.delete_episode(0);
  EQUAL(se.get_next_index(),episode_len - 1ul);
  EQUAL(se.get_stored_size(),episode_len - 1ul);
  EQUAL(se.get_stored_episode_size(),1ul);
}

int main(){

  test_DimensionalBuffer();
  test_PrioritizedSampler();
  test_SelectiveEnvironment();

  return 0;
}
