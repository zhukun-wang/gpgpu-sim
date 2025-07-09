// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda, George L. Yuan,
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "dram_sched.h"
#include "../abstract_hardware_model.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"

frfcfs_scheduler::frfcfs_scheduler(const memory_config *config, dram_t *dm,
                                   memory_stats_t *stats) {
  m_config = config;
  m_stats = stats;
  m_num_pending = 0;
  m_num_write_pending = 0;
  m_dram = dm;

  m_reorder_state = REORDER_READ_PHASE;
  m_batch_left    = REORDER_BATCH;

  m_queue = new std::list<dram_req_t *>[m_config->nbk];
  m_bins = new std::map<
      unsigned, std::list<std::list<dram_req_t *>::iterator> >[m_config->nbk];
  m_last_row =
      new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk];
  curr_row_service_time = new unsigned[m_config->nbk];
  row_service_timestamp = new unsigned[m_config->nbk];
  for (unsigned i = 0; i < m_config->nbk; i++) {
    m_queue[i].clear();
    m_bins[i].clear();
    m_last_row[i] = NULL;
    curr_row_service_time[i] = 0;
    row_service_timestamp[i] = 0;
  }
  if (m_config->seperate_write_queue_enabled) {
    m_write_queue = new std::list<dram_req_t *>[m_config->nbk];
    m_write_bins = new std::map<
        unsigned, std::list<std::list<dram_req_t *>::iterator> >[m_config->nbk];
    m_last_write_row =
        new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk];

    for (unsigned i = 0; i < m_config->nbk; i++) {
      m_write_queue[i].clear();
      m_write_bins[i].clear();
      m_last_write_row[i] = NULL;
    }
  }
  m_mode = READ_MODE;
}

void frfcfs_scheduler::add_req(dram_req_t *req) {

  req->sched_enqueue_cycle = m_dram->m_gpu->gpu_sim_cycle + m_dram->m_gpu->gpu_tot_sim_cycle;

  if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    assert(m_num_write_pending < m_config->gpgpu_frfcfs_dram_write_queue_size);
    m_num_write_pending++;
    m_write_queue[req->bk].push_front(req);
    std::list<dram_req_t *>::iterator ptr = m_write_queue[req->bk].begin();
    m_write_bins[req->bk][req->row].push_front(ptr);  // newest reqs to the
                                                      // front
  } else {
    assert(m_num_pending < m_config->gpgpu_frfcfs_dram_sched_queue_size);
    m_num_pending++;
    m_queue[req->bk].push_front(req);
    std::list<dram_req_t *>::iterator ptr = m_queue[req->bk].begin();
    m_bins[req->bk][req->row].push_front(ptr);  // newest reqs to the front
  }
}

void frfcfs_scheduler::data_collection(unsigned int bank) {
  if (m_dram->m_gpu->gpu_sim_cycle > row_service_timestamp[bank]) {
    curr_row_service_time[bank] =
        m_dram->m_gpu->gpu_sim_cycle - row_service_timestamp[bank];
    if (curr_row_service_time[bank] >
        m_stats->max_servicetime2samerow[m_dram->id][bank])
      m_stats->max_servicetime2samerow[m_dram->id][bank] =
          curr_row_service_time[bank];
  }
  curr_row_service_time[bank] = 0;
  row_service_timestamp[bank] = m_dram->m_gpu->gpu_sim_cycle;
  if (m_stats->concurrent_row_access[m_dram->id][bank] >
      m_stats->max_conc_access2samerow[m_dram->id][bank]) {
    m_stats->max_conc_access2samerow[m_dram->id][bank] =
        m_stats->concurrent_row_access[m_dram->id][bank];
  }
  m_stats->concurrent_row_access[m_dram->id][bank] = 0;
  m_stats->num_activates[m_dram->id][bank]++;
}

dram_req_t* frfcfs_scheduler::schedule(unsigned bank, unsigned curr_row) {
 
    unsigned n_banks = m_config->nbk; 
    m_last_is_write.resize(n_banks, false);
    m_last_valid.resize(n_banks, false);     

    std::list<dram_req_t*>* m_current_queue = m_queue;
    std::map<unsigned, std::list<std::list<dram_req_t*>::iterator>>* m_current_bins = m_bins;

    if (m_config->seperate_write_queue_enabled) {
        if (m_mode == READ_MODE && m_num_write_pending >= m_config->write_high_watermark)
            m_mode = WRITE_MODE;
        else if (m_mode == WRITE_MODE && m_num_write_pending <= m_config->write_low_watermark)
            m_mode = READ_MODE;
    }

    if (m_mode == WRITE_MODE) {
        m_current_queue = m_write_queue;
        m_current_bins = m_write_bins;
    }

    if (m_current_queue[bank].empty()) return NULL;

    auto match_type = [this](dram_req_t* req, unsigned bank) {
    	if (!m_last_valid[bank]) return true;          
    	bool is_write = req->data->is_write();
    	return is_write == m_last_is_write[bank];   
    };

    // Step 1: Row Hit + Same Type
    auto bin_it = m_current_bins[bank].find(curr_row);
    if (bin_it != m_current_bins[bank].end()) {
        for (auto it = bin_it->second.begin(); it != bin_it->second.end(); ++it) {
            dram_req_t* req = **it;
            if (match_type(req, bank)) {
                dram_req_t* result = req;
                m_current_queue[bank].erase(*it);
                bin_it->second.erase(it);
                if (bin_it->second.empty()) {
                    m_current_bins[bank].erase(curr_row);
                }
                
		update_counters(result, bank);
		
		FILE *f = fopen("dram_delay_log.txt", "a");

  fprintf(f,  "Clock: %8llu | DRAM: %2u | Bank: %2u | Row: %4u | Dispatch: %s | Addr: 0x%llx | Row Hit + Same Type\n",
                    m_dram->m_gpu->gpu_sim_cycle + m_dram->m_gpu->gpu_tot_sim_cycle,
                    m_dram->id,
                    result->bk,
                    result->row,
		    result->rw == READ ? "READ " : "WRITE",
                    result->addr);

  fclose(f);


                return result;
            }
        }

        // Step 2: Row Hit + Diff Type
        if (!bin_it->second.empty()) {
    	auto row_it = bin_it->second.begin();
    	dram_req_t* result = **row_it;
    	m_current_queue[bank].erase(*row_it);
    	bin_it->second.erase(row_it);
    	if (bin_it->second.empty()) {
            m_current_bins[bank].erase(curr_row);
    	}
   
        update_counters(result, bank);

    	FILE *f = fopen("dram_delay_log.txt", "a");

  	fprintf(f,  "Clock: %8llu | DRAM: %2u | Bank: %2u | Row: %4u | Dispatch: %s | Addr: 0x%llx | Row Hit + Diff Type\n",
                    m_dram->m_gpu->gpu_sim_cycle + m_dram->m_gpu->gpu_tot_sim_cycle,
                    m_dram->id,
                    result->bk,
                    result->row,
                    result->rw == READ ? "READ " : "WRITE",
                    result->addr);

  	fclose(f);

    	return result;
    	}
    }

    // Step 3: Oldest Same Type in Queue
    for (auto it = m_current_queue[bank].begin(); it != m_current_queue[bank].end(); ++it) {
        dram_req_t* req = *it;
        if (match_type(req, bank)) {
            auto row = req->row;
            auto bin_ptr = m_current_bins[bank].find(row);
            if (bin_ptr != m_current_bins[bank].end()) {
                for (auto row_it = bin_ptr->second.begin(); row_it != bin_ptr->second.end(); ++row_it) {
                    dram_req_t* row_req = **row_it;
                    if (match_type(row_req, bank)) {
                        dram_req_t* result = row_req;
                        m_current_queue[bank].erase(*row_it);
                        bin_ptr->second.erase(row_it);
                        if (bin_ptr->second.empty()) {
                            m_current_bins[bank].erase(row);
                        }
			update_counters(result, bank);
			
			FILE *f = fopen("dram_delay_log.txt", "a");

			fprintf(f,  "Clock: %8llu | DRAM: %2u | Bank: %2u | Row: %4u | Dispatch: %s | Addr: 0x%llx | Row Miss + Same Type\n",
                    		m_dram->m_gpu->gpu_sim_cycle + m_dram->m_gpu->gpu_tot_sim_cycle,
                    		m_dram->id,
                    		result->bk,
                    		result->row,
                    		result->rw == READ ? "READ " : "WRITE",
                    		result->addr);

  			fclose(f);


                        return result;
                    }
                }
            }
        }
    }

    // Step 4: Fallback - Oldest Request (Any Type)
if (!m_current_queue[bank].empty()) {
    auto head_it  = m_current_queue[bank].begin();
    dram_req_t* req = *head_it;
    unsigned row = req->row;

    auto bin_ptr = m_current_bins[bank].find(row);
    if (bin_ptr != m_current_bins[bank].end() && !bin_ptr->second.empty()) {
        auto row_it   = bin_ptr->second.begin();
        auto queue_it = *row_it;              
        dram_req_t* result = **row_it;

        m_current_queue[bank].erase(queue_it);   
        bin_ptr->second.erase(row_it);         
        if (bin_ptr->second.empty())
            m_current_bins[bank].erase(row);

	update_counters(result, bank);

        FILE* f = fopen("dram_delay_log.txt","a");
        fprintf(f, "Clock: %8llu | DRAM: %2u | Bank: %2u | Row: %4u | Dispatch: %s | Addr: 0x%llx | Row Miss + Diff Type\n",
                m_dram->m_gpu->gpu_sim_cycle + m_dram->m_gpu->gpu_tot_sim_cycle,
                m_dram->id, result->bk, result->row,
                result->rw==READ ? "READ ":"WRITE",
                result->addr);
        fclose(f);

        return result;
    }
}
/*
if (!m_current_queue[bank].empty()) {
    dram_req_t* any = m_current_queue[bank].front();
    m_current_queue[bank].pop_front();
    if (any->data->is_write()) m_num_write_pending--; else m_num_pending--;
    return any;
}
*/
    // If no request found (should not happen), return NULL
    return NULL;
}

void frfcfs_scheduler::update_counters(dram_req_t* req, unsigned bank) {
    bool is_write = req->data->is_write();
    if (m_config->seperate_write_queue_enabled) {
        if (is_write) m_num_write_pending--;
        else m_num_pending--;
    } else {
        m_num_pending--;
    }

    m_last_is_write[bank] = is_write;
    m_last_valid[bank]    = true;
}

void frfcfs_scheduler::print(FILE *fp) {
  for (unsigned b = 0; b < m_config->nbk; b++) {
    printf(" %u: queue length = %u\n", b, (unsigned)m_queue[b].size());
  }
}

void dram_t::scheduler_frfcfs() {
  unsigned mrq_latency;
  frfcfs_scheduler *sched = m_frfcfs_scheduler;
  while (!mrqq->empty()) {
    dram_req_t *req = mrqq->pop();

    // Power stats
    // if(req->data->get_type() != READ_REPLY && req->data->get_type() !=
    // WRITE_ACK)
    m_stats->total_n_access++;

    if (req->data->get_type() == WRITE_REQUEST) {
      m_stats->total_n_writes++;
    } else if (req->data->get_type() == READ_REQUEST) {
      m_stats->total_n_reads++;
    }

    req->data->set_status(IN_PARTITION_MC_INPUT_QUEUE,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    sched->add_req(req);
  }

  dram_req_t *req;
  unsigned i;
  for (i = 0; i < m_config->nbk; i++) {
    unsigned b = (i + prio) % m_config->nbk;
    if (!bk[b]->mrq) {
      req = sched->schedule(b, bk[b]->curr_row);

      if (req) {
        req->data->set_status(IN_PARTITION_MC_BANK_ARB_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        prio = (prio + 1) % m_config->nbk;
        bk[b]->mrq = req;
        if (m_config->gpgpu_memlatency_stat) {
          mrq_latency = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
                        bk[b]->mrq->timestamp;
          m_stats->tot_mrq_latency += mrq_latency;
          m_stats->tot_mrq_num++;
          bk[b]->mrq->timestamp =
              m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
          m_stats->mrq_lat_table[LOGB2(mrq_latency)]++;
          if (mrq_latency > m_stats->max_mrq_latency) {
            m_stats->max_mrq_latency = mrq_latency;
          }
        }

        break;
      }
    }
  }
}
