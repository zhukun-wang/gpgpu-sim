#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>

struct pf_entry_t {
    unsigned long long addr = 0;     
    bool     ready = false;
};

struct match_result_t {
    bool matched = false;
    bool all_ready  = true; 
};

class pf_table_t {
public:
    std::vector<pf_entry_t> table;

    void insert_inflight(unsigned long long addr) {
        pf_entry_t e;
        e.addr   = addr;
        e.ready  = false;
        table.push_back(e);
    }

    void mark_ready(unsigned long long addr) {
        for (auto &e : table) {
            if (match_exact(e, addr)) {
                e.ready = true;
            }
        }
    }

    match_result_t match_and_consume(unsigned long long addr) {
        match_result_t res;

	for (auto it = table.begin(); it != table.end(); ++it) {
		if (it->addr == addr) {
			res.matched = true;
			res.all_ready = it->ready;
			if (res.all_ready) {
				table.erase(it); 
			}
			break;  
		}
	}
	return res;
    }

private:
    static bool match_exact(const pf_entry_t &e, unsigned long long addr) {
        return e.addr == addr;
    }
};

