// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <list>
#include <set>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../statwrapper.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-sim.h"
#include "histogram.h"
#include "hashing.h"
#include "l2cache.h"
#include "l2cache_trace.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "mc_cache.h"
#include "oracle.h"

mem_fetch *partition_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr, unsigned long long cycle,
                                         unsigned long long streamID) const {
  assert(wr);
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID, WRITE_PACKET_SIZE, -1,
                                -1, -1, m_memory_config, cycle);
  return mf;
}

mem_fetch *partition_mf_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID,
                                wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid,
                                sid, tpc, m_memory_config, cycle, original_mf);
  return mf;
}
memory_partition_unit::memory_partition_unit(unsigned partition_id,
                                             const memory_config *config,
                                             class memory_stats_t *stats,
                                             class gpgpu_sim *gpu)
    : m_id(partition_id),
      m_config(config),
      m_stats(stats),
      m_arbitration_metadata(config),
      m_gpu(gpu) {
  m_dram = new dram_t(m_id, m_config, m_stats, this, gpu);

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    m_sub_partition[p] =
        new memory_sub_partition(sub_partition_id, m_config, stats, gpu);
  }

  m_prefetch_global_queue = new fifo_pipeline<mem_fetch>("PREFETCH-to-DRAM", 0, 64);

  m_bank_inflight.assign(m_config->nbk, 0);
  m_bank_row_pending.resize(m_config->nbk);

  //m_oracle.load("/accel-sim/accel-sim-framework/oracle_trace_llama.txt", 200);

  m_rlb_head = 0;
  for (unsigned i = 0; i < RLB_SIZE; ++i) {
    m_recent_lines[i].valid = false;
    m_recent_lines[i].line_addr = 0;
  }


  for (unsigned i = 0; i < ACTIVE_BASE_TABLE_SIZE; ++i) {
    m_active_base_table[i].valid = false;
    m_active_base_table[i].base = 0;
    m_active_base_table[i].last_addr = 0;
    m_active_base_table[i].timestamp = 0;
  }

   FILE* f = fopen("/accel-sim/accel-sim-framework/mpool.txt", "r");

    unsigned long long t;
    unsigned long long a;
    unsigned long long b;

    while (fscanf(f, "%llu %llx %llx", &t, &b, &a) == 3) {
        PrefetchMemEntry entry;
        //entry.time = t;
        entry.base = (uint64_t)b;
	entry.addr = (uint64_t)a;

	addrdec_t tlx;
	m_config->m_address_mapping.addrdec_tlx(a, &tlx);

        if (tlx.chip == m_id) {
          g_prefetch_mem_table.push_back(entry);
	}
    }

        //FILE *p = fopen("count1.txt", "a");
        //fprintf(p, "ID: %u Number: %u\n", m_id, g_prefetch_mem_table.size());
        //fclose(p);


    fclose(f);

    m_prefetch_template = nullptr;

}

void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  std::string mystring = mask.to_string<char, std::string::traits_type,
                                        std::string::allocator_type>();
  MEMPART_DPRINTF(
      "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
      "global_subpart=%u, sector_mask=%s \n",
      addr, p, global_subpart_id, mystring.c_str());
  m_sub_partition[p]->force_l2_tag_update(
      addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
}

memory_partition_unit::~memory_partition_unit() {
	
  delete m_dram;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p];
  }
  delete[] m_sub_partition;

  delete m_prefetch_global_queue;
  
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config)
    : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
      m_shared_credit(0) {
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
  }
  assert(m_shared_credit_limit >= 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true;
  } else {
    return false;
  }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1;
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
  }
  m_last_borrower = spid;
}

void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1;
  } else {
    m_shared_credit -= 1;
  }
  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = ");
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]);
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
  bool busy = false;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true;
    }
  }
  return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle);
  }
}

void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
  m_dram->visualizer_print(visualizer_file);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->visualizer_print(visualizer_file);
  }
}

// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

  MEMPART_DPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');

  return (has_dram_resource && !sub_partition_contention);
}

int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
}

void memory_partition_unit::simple_dram_model_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle)) {
    mem_fetch *mf_return = m_dram_latency_queue.front().req;
    if (mf_return->get_access_type() != L1_WRBK_ACC &&
        mf_return->get_access_type() != L2_WRBK_ACC) {
      mf_return->set_reply();

      unsigned dest_global_spid = mf_return->get_sub_partition_id();
      int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
      assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
      if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        if (mf_return->get_access_type() == L1_WRBK_ACC) {
          m_sub_partition[dest_spid]->set_done(mf_return);
          delete mf_return;
        } else {
	  m_stats->DRAM_to_L2_bytes += mf_return->get_data_size();
	
          m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
          mf_return->set_status(
              IN_PARTITION_DRAM_TO_L2_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_arbitration_metadata.return_credit(dest_spid);
          MEMPART_DPRINTF(
              "mem_fetch request %p return from dram to sub partition %d\n",
              mf_return, dest_spid);
        }
        m_dram_latency_queue.pop_front();
      }

    } else {
      this->set_done(mf_return);
      delete mf_return;
      m_dram_latency_queue.pop_front();
    }
  }

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;
      m_stats->L2_to_DRAM_bytes += mf->get_data_size();

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}
}

void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  m_stats->report_throughput(m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);

  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    if (mf_return->is_prefetch()) {

	 const new_addr_type la = mf_return->get_addr();   
	 pf_mark_arrived(la);

	 //FILE *f = fopen("count1.txt", "a");
	 //fprintf(f, "[Prefetch Arrive]0x%llx\n", la);
	 //fclose(f);

	 auto it = m_sram_unready.find(la);
	 if (it != m_sram_unready.end()) {

	 //FILE *f = fopen("count1.txt", "a");
         //fprintf(f, "[Unready Match]0x%llx\n", la);
         //fclose(f);


	   mem_fetch* w = it->second;

	   unsigned dest_global_spid = w->get_sub_partition_id();
    	   int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
           assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    	   if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {

        	m_stats->DRAM_to_L2_bytes += w->get_data_size();

        	m_sub_partition[dest_spid]->dram_L2_queue_push(w);
        	w->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        	MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n",
              	w, dest_spid);
		FILE *f = fopen("count.txt", "a");
                fprintf(f, "%llu\n", m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle - w->dram_entry_time);
                fclose(f);

		m_sram_unready.erase(it);
		m_dram->return_queue_pop();

		int b = bank_id_from_mf(mf_return);
		unsigned row = mf_return->get_tlx_addr().row;

		if (m_bank_inflight[b] > 0) --m_bank_inflight[b];
		auto it = m_bank_row_pending[b].find(row);
		if (it != m_bank_row_pending[b].end()) {
    			if (--it->second == 0)
        			m_bank_row_pending[b].erase(it);
		}

		m_stats->DRAM_to_L2_bytes += mf_return->get_data_size();
		delete mf_return;
	   } else {
		sram_delay_t s;
		s.req = w;
		s.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
		m_sram_ready.push_back(s);

		m_sram_unready.erase(it);
		m_dram->return_queue_pop();

		int b = bank_id_from_mf(mf_return);
		unsigned row = mf_return->get_tlx_addr().row;

		if (m_bank_inflight[b] > 0) --m_bank_inflight[b];
		auto it = m_bank_row_pending[b].find(row);
		if (it != m_bank_row_pending[b].end()) {
			if (--it->second == 0)
				m_bank_row_pending[b].erase(it);
		}
		m_stats->DRAM_to_L2_bytes += mf_return->get_data_size();
		delete mf_return;
	   }

	   if (!m_sram_ready.empty() && ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >= m_sram_ready.front().ready_cycle))  {
	     unsigned pf_dest_global_spid = m_sram_ready.front().req->get_sub_partition_id();
	     int pf_dest_spid = global_sub_partition_id_to_local_id(pf_dest_global_spid);
	     assert(m_sub_partition[pf_dest_spid]->get_id() == pf_dest_global_spid);
	     if (pf_dest_spid != dest_spid) {
	        mem_fetch *mf = m_sram_ready.front().req;

		if (!m_sub_partition[pf_dest_spid]->dram_L2_queue_full()) {
			m_stats->DRAM_to_L2_bytes += mf->get_data_size();

			m_sub_partition[pf_dest_spid]->dram_L2_queue_push(mf);
			mf->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
			MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf, pf_dest_spid);
			FILE *f = fopen("count.txt", "a");
                        fprintf(f, "%llu\n", m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle - mf->dram_entry_time);
                        fclose(f);
			m_sram_ready.pop_front();
		}
	     }
	   }
	 } else {
	   m_dram->return_queue_pop();
	   int b = bank_id_from_mf(mf_return);
	   unsigned row = mf_return->get_tlx_addr().row;

	   if (m_bank_inflight[b] > 0) --m_bank_inflight[b];
	   auto it = m_bank_row_pending[b].find(row);
	   if (it != m_bank_row_pending[b].end()) {
		   if (--it->second == 0)
			   m_bank_row_pending[b].erase(it);
	   }

	   m_stats->DRAM_to_L2_bytes += mf_return->get_data_size();
	   delete mf_return;
	 }
    } else {
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        m_sub_partition[dest_spid]->set_done(mf_return);
	int b = bank_id_from_mf(mf_return);
	unsigned row = mf_return->get_tlx_addr().row;

	if (m_bank_inflight[b] > 0) --m_bank_inflight[b];
	auto it = m_bank_row_pending[b].find(row);
	if (it != m_bank_row_pending[b].end()) {
		if (--it->second == 0)
			m_bank_row_pending[b].erase(it);
	}
        delete mf_return;
      } else {
	m_stats->DRAM_to_L2_bytes += mf_return->get_data_size();

        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.return_credit(dest_spid);
        MEMPART_DPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
	FILE *f = fopen("count.txt", "a");
        fprintf(f, "%llu\n", m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle - mf_return->dram_entry_time);
        fclose(f);
	int b = bank_id_from_mf(mf_return);
	unsigned row = mf_return->get_tlx_addr().row;

	if (m_bank_inflight[b] > 0) --m_bank_inflight[b];
	auto it = m_bank_row_pending[b].find(row);
	if (it != m_bank_row_pending[b].end()) {
		if (--it->second == 0)
			m_bank_row_pending[b].erase(it);
	}
      }
      m_dram->return_queue_pop();
    }

    if (!m_sram_ready.empty() && ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >= m_sram_ready.front().ready_cycle))  {
	    unsigned pf_dest_global_spid = m_sram_ready.front().req->get_sub_partition_id();
	    int pf_dest_spid = global_sub_partition_id_to_local_id(pf_dest_global_spid);
	    assert(m_sub_partition[pf_dest_spid]->get_id() == pf_dest_global_spid);
	    if (pf_dest_spid != dest_spid) {
		    mem_fetch *mf = m_sram_ready.front().req;

		    if (!m_sub_partition[pf_dest_spid]->dram_L2_queue_full()) {
			    m_stats->DRAM_to_L2_bytes += mf->get_data_size();

			    m_sub_partition[pf_dest_spid]->dram_L2_queue_push(mf);
			    mf->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
			    MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf, pf_dest_spid);
			    FILE *f = fopen("count.txt", "a");
			    fprintf(f, "%llu\n", m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle - mf->dram_entry_time);
			    fclose(f);
			    m_sram_ready.pop_front();
		    }
	    }
    }
    }
  } else {
    if (!m_sram_ready.empty() && ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >= m_sram_ready.front().ready_cycle))  {
	    mem_fetch *mf = m_sram_ready.front().req;
	    unsigned pf_dest_global_spid = mf->get_sub_partition_id();
	    int pf_dest_spid = global_sub_partition_id_to_local_id(pf_dest_global_spid);
	    assert(m_sub_partition[pf_dest_spid]->get_id() == pf_dest_global_spid);
	    if (!m_sub_partition[pf_dest_spid]->dram_L2_queue_full()) {
		    m_stats->DRAM_to_L2_bytes += mf->get_data_size();

		    m_sub_partition[pf_dest_spid]->dram_L2_queue_push(mf);
		    mf->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
		    MEMPART_DPRINTF("mem_fetch request %p return from dram to sub partition %d\n", mf, pf_dest_spid);
		    FILE *f = fopen("count.txt", "a");
		    fprintf(f, "%llu\n", m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle - mf->dram_entry_time);
		    fclose(f);
		    m_sram_ready.pop_front();
	    }
    }
    m_dram->return_queue_pop();
  }

  m_dram->cycle();
  m_dram->dram_log(SAMPLELOG);

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  
  bool issued = false;

  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;
      
      if (mf->is_write()){
      	m_stats->L2_to_DRAM_bytes += mf->get_data_size();
      }

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);

      mf->dram_entry_time = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;

      const new_addr_type la = mf->get_addr();

      active_mpool();

      for (auto it = mpool.begin(); it != mpool.end(); ) {
        if (it->addr == la) {
            it = mpool.erase(it);
            break;
        } else {
            ++it;
        }
      }

      update_active_base_table(la);

      if (!mf->is_write() && pf_exists(la)) {

	 FILE *p = fopen("count1.txt", "a");
         fprintf(p, "Prefetch Hit\n");
         fclose(p);

         if (pf_is_arrived(la)) {
            sram_delay_t s;
	    s.req = mf;
	    s.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + 2;
	    m_sram_ready.push_back(s);
	    rlb_insert(la);
	    //break;

	  } else {
	    m_sram_unready.emplace(la, mf);
	  }
      } else {
        dram_delay_t d;
        d.req = mf;
        d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
        m_dram_latency_queue.push_back(d);
        mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.borrow_credit(spid);
	
	if (!mf->is_write()){
	    rlb_insert(mf->get_addr());
	}
	
	int b = bank_id_from_mf(mf);
	unsigned row = mf->get_tlx_addr().row;

  	++m_bank_inflight[b];
	m_bank_row_pending[b][row]++;

        issued = true;

        break;  // the DRAM should only accept one request per cycle
      }
    }
  }
  //}


  if (!issued) {

    if (!m_dram->full(false)) {

	mem_fetch* pf = generate_prefetch_after_issue();

	if(pf){

          dram_delay_t d;
          d.req = pf;
          d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                        m_config->dram_latency;
          m_dram_latency_queue.push_back(d);
          pf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);

	  int b = bank_id_from_mf(pf);
	  unsigned row = pf->get_tlx_addr().row;

	  rlb_insert(pf->get_addr());
	  ++m_bank_inflight[b];
	  m_bank_row_pending[b][row]++;
	}
    }
  }

  // DRAM latency queue
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    mem_fetch *mf = m_dram_latency_queue.front().req;
    m_dram_latency_queue.pop_front();
    m_dram->push(mf);
  }
}

void memory_partition_unit::set_done(mem_fetch *mf) {
  unsigned global_spid = mf->get_sub_partition_id();
  int spid = global_sub_partition_id_to_local_id(global_spid);
  assert(m_sub_partition[spid]->get_id() == global_spid);
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    m_arbitration_metadata.return_credit(spid);
    MEMPART_DPRINTF(
        "mem_fetch request %p return from dram to sub partition %d\n", mf,
        spid);
  }
  m_sub_partition[spid]->set_done(mf);
}

void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
    unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
}

void memory_partition_unit::print(FILE *fp) const {
  fprintf(fp, "Memory Partition %u: \n", m_id);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->print(fp);
  }
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          m_dram_latency_queue.size());
  for (std::list<dram_delay_t>::const_iterator mf_dlq =
           m_dram_latency_queue.begin();
       mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
    mem_fetch *mf = mf_dlq->req;
    fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
    if (mf)
      mf->print(fp);
    else
      fprintf(fp, " <NULL mem_fetch?>\n");
  }
  m_dram->print(fp);
}

memory_sub_partition::memory_sub_partition(unsigned sub_partition_id,
                                           const memory_config *config,
                                           class memory_stats_t *stats,
                                           class gpgpu_sim *gpu) {
  m_id = sub_partition_id;
  m_config = config;
  m_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;

  assert(m_id < m_config->m_n_mem_sub_partition);

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  m_L2interface = new L2interface(this);
  m_mf_allocator = new partition_mf_allocator(config);

  if (!m_config->m_L2_config.disabled())
    m_L2cache = new l2_cache(L2c_name, m_config->m_L2_config, -1, -1,
                             m_L2interface, m_mf_allocator,
                             IN_PARTITION_L2_MISS_QUEUE, gpu, L2_GPU_CACHE);

  unsigned int icnt_L2;
  unsigned int L2_dram;
  unsigned int dram_L2;
  unsigned int L2_icnt;
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  wb_addr = -1;

}

memory_sub_partition::~memory_sub_partition() {
  delete m_icnt_L2_queue;
  delete m_L2_dram_queue;
  delete m_dram_L2_queue;
  delete m_L2_icnt_queue;
  delete m_L2cache;
  delete m_L2interface;
}

void memory_sub_partition::cache_cycle(unsigned cycle) {
  // L2 fill responses
  if (!m_config->m_L2_config.disabled()) {
    if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
      mem_fetch *mf = m_L2cache->next_access();
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                            // upper level cache
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2_icnt_queue->push(mf);
      } else {
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          assert(original_wr_mf);
          original_wr_mf->set_reply();
          original_wr_mf->set_status(
              IN_PARTITION_L2_TO_ICNT_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_L2_icnt_queue->push(original_wr_mf);
        }
        m_request_tracker.erase(mf);
        delete mf;
      }
    }
  }

  // DRAM to L2 (texture) and icnt (not texture)
  if (!m_dram_L2_queue->empty()) {
    mem_fetch *mf = m_dram_L2_queue->top();
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      if (m_L2cache->fill_port_free()) {
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        m_dram_L2_queue->pop();
      }
    } else if (!m_L2_icnt_queue->full()) {
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_icnt_queue->push(mf);
      m_dram_L2_queue->pop();
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();

  // new L2 texture accesses and/or non-texture accesses
  if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
    mem_fetch *mf = m_icnt_L2_queue->top();
    if (!m_config->m_L2_config.disabled() &&
        ((m_config->m_L2_texure_only && mf->istexture()) ||
         (!m_config->m_L2_texure_only))) {
      // L2 is enabled and access is for L2
      bool output_full = m_L2_icnt_queue->full();
      bool port_free = m_L2cache->data_port_free();
      if (!output_full && port_free) {
        std::list<cache_event> events;
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        bool write_sent = was_write_sent(events);
        bool read_sent = was_read_sent(events);
        MEM_SUBPART_DPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                            mf->get_addr(), status);

        if (status == HIT) {
          if (!write_sent) {
            // L2 cache replies
            assert(!read_sent);
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
            m_icnt_L2_queue->pop();
          } else {
            assert(write_sent);
            m_icnt_L2_queue->pop();
          }
        } else if (status != RESERVATION_FAIL) {
          if (mf->is_write() &&
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else if (m_config->m_L2_config.get_write_policy() == WRITE_BACK) {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
          }
          // L2 cache accepted request
          m_icnt_L2_queue->pop();
        } else {
          assert(!write_sent);
          assert(!read_sent);
          // L2 cache lock-up: will try again next cycle
        }
      }
    } else {
      // L2 is disabled or non-texture access to texture-only L2
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_dram_queue->push(mf);
      m_icnt_L2_queue->pop();
    }
  }

  // ROP delay queue
  if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
      !m_icnt_L2_queue->full()) {
    mem_fetch *mf = m_rop.front().req;
    m_rop.pop();
    m_icnt_L2_queue->push(mf);
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }
}

bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_avilable_size(size);
}

bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  m_dram_L2_queue->push(mf);
}

void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout;
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
}

void memory_sub_partition::print(FILE *fp) const {
  if (!m_request_tracker.empty()) {
    fprintf(fp, "Memory Sub Parition %u: pending memory requests:\n", m_id);
    for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
         r != m_request_tracker.end(); ++r) {
      mem_fetch *mf = *r;
      if (mf)
        mf->print(fp);
      else
        fprintf(fp, " <NULL mem_fetch?>\n");
    }
  }
  if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print(gzFile visualizer_file) {
  gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
  gzprintf(visualizer_file, "Ltwowritehit: %d\n", L2_write_hit);
  gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
  gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
  clear_L2_stats_pw();

  if (num_mfs)
    gzprintf(visualizer_file, "averagemflatency: %lld\n",
             mf_total_lat / num_mfs);
}

void memory_stats_t::clear_L2_stats_pw() {
  L2_write_miss = 0;
  L2_write_hit = 0;
  L2_read_miss = 0;
  L2_read_hit = 0;
}

void gpgpu_sim::print_dram_stats(FILE *fout) const {
  unsigned cmd = 0;
  unsigned activity = 0;
  unsigned nop = 0;
  unsigned act = 0;
  unsigned pre = 0;
  unsigned rd = 0;
  unsigned wr = 0;
  unsigned wr_WB = 0;
  unsigned req = 0;
  unsigned tot_cmd = 0;
  unsigned tot_nop = 0;
  unsigned tot_act = 0;
  unsigned tot_pre = 0;
  unsigned tot_rd = 0;
  unsigned tot_wr = 0;
  unsigned tot_req = 0;

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i]->set_dram_power_stats(cmd, activity, nop, act,
                                                     pre, rd, wr, wr_WB, req);
    tot_cmd += cmd;
    tot_nop += nop;
    tot_act += act;
    tot_pre += pre;
    tot_rd += rd;
    tot_wr += wr + wr_WB;
    tot_req += req;
  }
  fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
  fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
  fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
  fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
  fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
  fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
  fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
}

unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush();
  }
  return 0;  // TODO: write the flushed data to the main memory
}

unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate();
  }
  return 0;
}

bool memory_sub_partition::busy() const { return !m_request_tracker.empty(); }

std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    mem_access_byte_mask_t mask;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
    // This is for constant cache
  } else if (mf->get_data_size() == 64 &&
             (mf->get_access_sector_mask().all() ||
              mf->get_access_sector_mask().none())) {
    unsigned start;
    if (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0)
      start = 0;
    else
      start = 2;
    mem_access_byte_mask_t mask;
    for (unsigned i = start; i < start + 2; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
          mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
  } else {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf,
            mf->get_streamID());

        result.push_back(n_mf);
      }
    }
  }
  if (result.size() == 0) assert(0 && "no mf sent");
  return result;
}

void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_stats->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
    else
      reqs.push_back(m_req);

    for (unsigned i = 0; i < reqs.size(); ++i) {
      mem_fetch *req = reqs[i];
      m_request_tracker.insert(req);
      if (req->istexture()) {
        m_icnt_L2_queue->push(req);
        req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      } else {
        rop_delay_t r;
        r.req = req;
        r.ready_cycle = cycle + m_config->rop_latency;
        m_rop.push(r);
        req->set_status(IN_PARTITION_ROP_DELAY,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      }
    }
  }
}

mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {
  m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
  }
}

void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
  }
}

void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
  }
}

void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
  }
}

void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
  // Support for L2 AerialVision stats
  // Per-sub-partition stats would be trivial to extend from this
  cache_sub_stats_pw temp_sub_stats;
  get_L2cache_sub_stats_pw(temp_sub_stats);

  m_stats->L2_read_miss += temp_sub_stats.read_misses;
  m_stats->L2_write_miss += temp_sub_stats.write_misses;
  m_stats->L2_read_hit += temp_sub_stats.read_hits;
  m_stats->L2_write_hit += temp_sub_stats.write_hits;

  clear_L2cache_stats_pw();
}

mem_fetch* memory_partition_unit::new_prefetch_req(new_addr_type addr, mem_fetch* original) {

    mem_fetch* new_mf = new mem_fetch(original->get_access(),
                                  //&original->get_inst(),
				  NULL,
                                  original->get_streamID(),
                                  original->get_ctrl_size(),
                                  original->get_wid(),
                                  original->get_sid(),
                                  original->get_tpc(),
                                  original->get_mem_config(),
                                  m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
                                  original->get_original_mf(),
                                  original->get_original_wr_mf());

    addrdec_t tlx;
    m_config->m_address_mapping.addrdec_tlx(addr, &tlx);
    new_mf->change_addr(tlx);
    new_mf->set_addr(addr);

    new_mf->mark_prefetch();

    return new_mf;
}

mem_fetch* memory_partition_unit::generate_prefetch_after_issue() {


    uint64_t pf_addr = pick_prefetch_addr_from_pattern();

    if (pf_addr == 0) return nullptr;

    for (auto it = mpool.begin(); it != mpool.end(); ) {
        if (it->addr == pf_addr) {
            it = mpool.erase(it);  
            break;           
        } else {
            ++it;
        }
    }

    mem_fetch* pf = new_prefetch_req(pf_addr, m_prefetch_template);

    pf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);

    pf_track_request(pf_addr);

    //FILE *p = fopen("count1.txt", "a");
    //fprintf(p, "Prefetch: ID: %u Pool Rest: %u\n", m_id, mpool.size());
    //fclose(p);


    return pf;

}

void memory_partition_unit::active_mpool() {

    auto &rec = mrecord;

    for (auto rec_it = rec.begin(); rec_it != rec.end(); ) {

        new_addr_type b = *rec_it;

        for (auto it = g_prefetch_mem_table.begin();
             it != g_prefetch_mem_table.end(); ) {

            if (it->base == b) {

		addrdec_t tlx;
                m_config->m_address_mapping.addrdec_tlx(it->addr, &tlx);

                int bk;
                switch (m_config->dram_bnk_indexing_policy) {
                    case LINEAR_BK_INDEX:
                        bk = tlx.bk;
                        break;
                    case BITWISE_XORING_BK_INDEX:
                        bk = bitwise_hash_function(tlx.row, tlx.bk, m_config->nbk);
                        assert(bk < (int)m_config->nbk);
                        break;
                    case IPOLY_BK_INDEX:
                        bk = ipoly_hash_function(tlx.row, tlx.bk, m_config->nbk);
                        assert(bk < (int)m_config->nbk);
                        break;
                    case CUSTOM_BK_INDEX:
                        bk = tlx.bk;
                        break;
                    default:
                        assert(0 && "Undefined bank index function.");
                        bk = 0;
                        break;
                }

                PoolCand cand;
                cand.addr = it->addr;
                cand.bank = bk;
                cand.row = tlx.row;
                mpool.push_back(cand);

                it = g_prefetch_mem_table.erase(it);
            } else {
                ++it;
            }
        }
        rec_it = rec.erase(rec_it);
    }
}

int memory_partition_unit::bank_id_from_mf(class mem_fetch* mf) {

	unsigned banks = m_config->nbk;
	const addrdec_t &tlx = mf->get_tlx_addr();
	int bk;

	switch (m_config->dram_bnk_indexing_policy) {
    	  case LINEAR_BK_INDEX: {
      	    bk = tlx.bk;
      	    break;
    	  }
    	  case BITWISE_XORING_BK_INDEX: {
      	    bk = bitwise_hash_function(tlx.row, tlx.bk, banks);
      	    assert(bk < banks);
      	    break;
    	  }
    	  case IPOLY_BK_INDEX: {
      	    bk = ipoly_hash_function(tlx.row, tlx.bk, banks);
      	    assert(bk < banks);
      	    break;
    	  }
    	  case CUSTOM_BK_INDEX:
      	    break;
    	  default:
      	    assert("\nUndefined bank index function.\n" && 0);
      	    break;
  	}

	return bk;

}

bool memory_partition_unit::rlb_contains(new_addr_type line){
    for (unsigned i = 0; i < RLB_SIZE; ++i) {
        if (m_recent_lines[i].valid && m_recent_lines[i].line_addr == line) {
            return true;
        }
    }
    return false;
}

void memory_partition_unit::rlb_insert(new_addr_type line) {
    for (unsigned i = 0; i < RLB_SIZE; ++i) {
        if (m_recent_lines[i].valid && m_recent_lines[i].line_addr == line) {
            return;
        }
    }

    m_recent_lines[m_rlb_head].line_addr = line;
    m_recent_lines[m_rlb_head].valid = true;
    m_rlb_head = (m_rlb_head + 1) % RLB_SIZE;
}

new_addr_type memory_partition_unit::pick_prefetch_addr_from_pattern()
{

    if (mpool.empty())
        return 0;

    auto has_pending_row = [&](int bk, unsigned row) -> bool {
        if ((unsigned)bk >= m_bank_row_pending.size()) return false;
        const auto &mp = m_bank_row_pending[bk];
        return mp.find(row) != mp.end();
    };

    for (const auto &c : mpool) {
        if ((unsigned)c.bank < m_bank_inflight.size() && m_bank_inflight[c.bank] == 0) {
            return c.addr;
        }
    }

    for (const auto &c : mpool) {
        if ((unsigned)c.bank < m_bank_inflight.size()
            && m_bank_inflight[c.bank] < 4
            && has_pending_row(c.bank, c.row)) {
            return c.addr;
        }
    }

    int bank_count = m_bank_inflight.size();

    int sorted_ids[16];  

    for (int i = 0; i < bank_count; ++i) {
        sorted_ids[i] = i;
    }

    for (int i = 0; i < bank_count - 1; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < bank_count; ++j) {
            if (m_bank_inflight[sorted_ids[j]] <
                m_bank_inflight[sorted_ids[min_idx]]) {
                min_idx = j;
            }
        }

        if (min_idx != i) {
            int tmp = sorted_ids[i];
            sorted_ids[i] = sorted_ids[min_idx];
            sorted_ids[min_idx] = tmp;
        }
    }

    for (int i = 0; i < bank_count; ++i) {
  
        int bk = sorted_ids[i];

        if (m_bank_inflight[bk] > 6)
            break;

        for (const auto &c : mpool) {
            if ((unsigned)c.bank == (unsigned)bk) {
                return c.addr;
            }
        }
    }

    return 0;

}

new_addr_type memory_partition_unit::align_active_base(new_addr_type addr)
{
    return addr & ~((new_addr_type)0xFFFF);
}

int memory_partition_unit::find_active_base_slot(new_addr_type base)
{
    for (unsigned i = 0; i < ACTIVE_BASE_TABLE_SIZE; ++i) {
        if (m_active_base_table[i].valid &&
            m_active_base_table[i].base == base) {
            return (int)i;
        }
    }
    return -1;
}

void memory_partition_unit::update_active_base_table(new_addr_type addr)
{
    new_addr_type base = align_active_base(addr);
    unsigned long long now = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;

    int idx = find_active_base_slot(base);
    if (idx >= 0) {
        m_active_base_table[idx].last_addr  = addr;
        m_active_base_table[idx].timestamp  = now;
        return;
    }

    int free_idx = -1;
    for (unsigned i = 0; i < ACTIVE_BASE_TABLE_SIZE; ++i) {
        if (!m_active_base_table[i].valid) {
            free_idx = i;
            break;
        }
    }

    if (free_idx >= 0) {
        m_active_base_table[free_idx].valid      = true;
        m_active_base_table[free_idx].base       = base;
        m_active_base_table[free_idx].last_addr  = addr;
        m_active_base_table[free_idx].timestamp  = now;
        return;
    }

    unsigned lru_idx = 0;
    unsigned long long lru_time = m_active_base_table[0].timestamp;
    for (unsigned i = 1; i < ACTIVE_BASE_TABLE_SIZE; ++i) {
        if (m_active_base_table[i].timestamp < lru_time) {
            lru_time = m_active_base_table[i].timestamp;
            lru_idx = i;
        }
    }

    m_active_base_table[lru_idx].valid      = true;
    m_active_base_table[lru_idx].base       = base;
    m_active_base_table[lru_idx].last_addr  = addr;
    m_active_base_table[lru_idx].timestamp  = now;
}

unsigned long long memory_partition_unit::now(){
  return m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
}

void memory_partition_unit::make_prefetch_from_template(mem_fetch* original) {

    m_prefetch_template = new mem_fetch(original->get_access(),
                                  //&original->get_inst(),
				  NULL,
                                  original->get_streamID(),
                                  original->get_ctrl_size(),
                                  original->get_wid(),
                                  original->get_sid(),
                                  original->get_tpc(),
                                  original->get_mem_config(),
                                  m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
                                  original->get_original_mf(),
                                  original->get_original_wr_mf());

    FILE *f = fopen("count1.txt", "a");
    fprintf(f, "Template Success\n");
    fclose(f);

}
