#include <vector>
#include <algorithm>

struct pf_entry_t {
	unsigned bk = 0;
	unsigned row = 0;
	unsigned col = 0;
	unsigned nbytes;
	bool ready;
};

struct match_result_t {
	bool full_cover;
	bool all_ready;
	//std::vector<pf_entry_t*> unready_list;
};

class pf_table_t {
public:
	std::vector<pf_entry_t> table;

	void insert_inflight(unsigned col, unsigned row, unsigned bk, unsigned nbytes) {
		pf_entry_t e;
		e.col = col;
		e.row = row;
		e.bk = bk;
		e.nbytes = nbytes;
		e.ready = false;
		table.push_back(e);
	}

	void mark_ready(unsigned col, unsigned row, unsigned bk, unsigned nbytes) {
		for (auto &e : table) {
			if (match_exact(e, col, row, bk, nbytes)) {
				e.ready = true;
			}
		}
	}

	match_result_t match_and_consume(unsigned col,
                                     unsigned row,
                                     unsigned bk,
				     unsigned nbytes) {
		match_result_t res{false, true};

		std::vector<std::pair<unsigned, unsigned>> hit_ranges;
		std::vector<pf_entry_t*> matched_entries;

		unsigned req_start_col = col;
		unsigned req_end_col = col + nbytes - 1;

		for (auto &e : table) {
			//FILE *f = fopen("count.txt", "a");
			//fprintf(f, "[Prefetching Request] Bank: %u Row: %u Col Start: %u Col End: %u\n", e.bk, e.row, e.col, e.col + e.nbytes - 1);
			//fprintf(f, "[Real Memory Request] Bank: %u Row: %u Col Start: %u Col End: %u\n", bk, row, req_start_col, req_end_col);
			//fclose(f);

			if (e.row != row || e.bk != bk) continue;

			unsigned pf_start = e.col;
			unsigned pf_end   = e.col + e.nbytes - 1;
			
			//FILE *f = fopen("count.txt", "a");
			//fprintf(f, "[Prefetching Request] Bank: %u Row: %u Col Start: %u Col End: %u\n", e.bk, e.row, pf_start, pf_end);
			//fprintf(f, "[Real Memory Request] Bank: %u Row: %u Col Start: %u Col End: %u\n", bk, row, req_start_col, req_end_col);
			//fclose(f);

			bool overlap = !(pf_end < req_start_col || pf_start > req_end_col);
			if (overlap) {
				hit_ranges.push_back({pf_start, pf_end});
				if (pf_end > req_end_col){
					matched_entries.push_back(&e);
				}
				if (!e.ready) {
					res.all_ready = false;
					//res.unready_list.push_back(&e);
				}
			}
		}

		if (hit_ranges.empty()) {
			return res;
		}

		std::sort(hit_ranges.begin(), hit_ranges.end());
		unsigned merge_start = hit_ranges[0].first;
		unsigned merge_end   = hit_ranges[0].second;

		for (size_t i = 1; i < hit_ranges.size(); ++i) {
			if (hit_ranges[i].first <= merge_end + 1) {
				merge_end = std::max(merge_end, hit_ranges[i].second);
            		} else {
				break;
			}
		}

		res.full_cover = (merge_start <= req_start_col && merge_end >= req_end_col);

		if (res.full_cover && res.all_ready) {
			for (auto *entry : matched_entries) {
				erase_entry(*entry);
			}
		}

		return res;
	}

	bool all_ready(const std::vector<pf_entry_t*> &entries) {
		for (auto *e : entries) {
			if (!e->ready) return false;
		}

		for (auto *e : entries) {
			erase_entry(*e);
		}
		return true;
	}

private:
	static bool match_exact(const pf_entry_t &e, unsigned col, unsigned row, unsigned bk, unsigned nbytes) {
        	return e.col == col && e.row == row && e.bk == bk && e.nbytes == nbytes;
    	}

	void erase_entry(const pf_entry_t &target) {
		table.erase(std::remove_if(table.begin(), table.end(),
					[&](const pf_entry_t &e) {
					return match_exact(e, target.col, target.row, target.bk, target.nbytes);
					}),
				table.end());
	}

};






