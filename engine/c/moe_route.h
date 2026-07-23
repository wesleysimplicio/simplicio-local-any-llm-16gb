#ifndef COLI_MOE_ROUTE_H
#define COLI_MOE_ROUTE_H

#include <math.h>
#include <string.h>

/*
 * Sigmoid/noaux_tc routing shared by GLM, DeepSeek-V3 and Kimi K2.
 *
 * Correction bias participates only in expert selection. Returned weights
 * are the unbiased sigmoid probabilities, matching the family reference
 * implementations. Scratch buffers are caller-owned to avoid decode-time
 * allocation.
 */
static inline void coli_moe_select(const float *router_logits, const float *correction_bias,
                                   int n_experts, int topk, int n_group, int topk_group,
                                   float *prob, float *choice, float *group_score,
                                   unsigned char *group_selected, int *indices, float *weights){
    for(int e=0;e<n_experts;e++){
        prob[e]=1.f/(1.f+expf(-router_logits[e]));
        choice[e]=prob[e]+correction_bias[e];
    }

    if(n_group>1){
        int group_size=n_experts/n_group;
        for(int g=0;g<n_group;g++){
            float top1=-1e30f, top2=-1e30f;
            for(int j=0;j<group_size;j++){
                float value=choice[g*group_size+j];
                if(value>top1){ top2=top1; top1=value; }
                else if(value>top2) top2=value;
            }
            group_score[g]=top1+top2;
        }
        memset(group_selected,0,(size_t)n_group);
        for(int rank=0;rank<topk_group;rank++){
            int best=-1; float best_value=-1e30f;
            for(int g=0;g<n_group;g++){
                if(!group_selected[g] && group_score[g]>best_value){
                    best_value=group_score[g]; best=g;
                }
            }
            if(best>=0) group_selected[best]=1;
        }
        for(int g=0;g<n_group;g++){
            if(group_selected[g]) continue;
            for(int j=0;j<group_size;j++) choice[g*group_size+j]=-INFINITY;
        }
    }

    for(int rank=0;rank<topk;rank++){
        int best=-1; float best_value=-1e30f;
        for(int e=0;e<n_experts;e++){
            int already_selected=0;
            for(int prior=0;prior<rank;prior++){
                if(indices[prior]==e){ already_selected=1; break; }
            }
            if(!already_selected && choice[e]>best_value){
                best_value=choice[e]; best=e;
            }
        }
        indices[rank]=best;
        weights[rank]=prob[best];
    }
}

#endif
