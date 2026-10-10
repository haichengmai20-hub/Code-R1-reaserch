"""Pure-Python two-logit policy-gradient lab; not torch, a tokenizer or Code-R1 training."""
import argparse
import json
import math
from pathlib import Path
import sys


def softmax(logits, temperature=1.):
    if temperature <= 0 or not logits or any(not math.isfinite(z) for z in logits):
        raise ValueError('Need finite logits and positive training temperature.')
    scaled=[z/temperature for z in logits];largest=max(scaled)
    weights=[math.exp(z-largest) for z in scaled];total=sum(weights)
    return [w/total for w in weights]


def policy_loss_and_gradient(logits, selected, old_logprob, advantage, cliprange=.2, temperature=1.):
    probabilities=softmax(logits,temperature)
    if type(selected) is not int or not 0 <= selected < len(logits) or not 0 < cliprange < 1:
        raise ValueError('Invalid selected token or clip range.')
    if not math.isfinite(old_logprob) or not math.isfinite(advantage):
        raise ValueError('Old logprob and advantage must be finite fixed values.')
    new_logprob=math.log(probabilities[selected]);ratio=math.exp(new_logprob-old_logprob)
    clipped=max(1-cliprange,min(1+cliprange,ratio))
    raw=-advantage*ratio;bounded=-advantage*clipped
    # At exact clip boundaries the derivative is not uniquely defined. Exercises avoid them.
    clipped_branch=bounded > raw
    loss=max(raw,bounded)
    gradient=[0.]*len(logits) if clipped_branch else [
        -advantage*ratio*((1. if j==selected else 0.)-p)/temperature
        for j,p in enumerate(probabilities)]
    return loss,gradient,{'probabilities':probabilities,'new_logprob':new_logprob,'old_logprob':old_logprob,
                         'ratio':ratio,'clipped_ratio':clipped,'clipped_branch':clipped_branch,
                         'advantage_fixed':advantage,'temperature':temperature}


def finite_difference(logits, selected, old_logprob, advantage, cliprange=.2, temperature=1., epsilon=1e-6):
    result=[]
    for j in range(len(logits)):
        positive=list(logits);negative=list(logits);positive[j]+=epsilon;negative[j]-=epsilon
        lp=policy_loss_and_gradient(positive,selected,old_logprob,advantage,cliprange,temperature)[0]
        lm=policy_loss_and_gradient(negative,selected,old_logprob,advantage,cliprange,temperature)[0]
        result.append((lp-lm)/(2*epsilon))
    return result


def masked_mean(values,masks):
    if len(values)!=len(masks) or any(len(v)!=len(m) for v,m in zip(values,masks)):
        raise ValueError('Values and masks must have identical shapes.')
    denominator=sum(sum(row) for row in masks)
    if denominator <= 0 or any(x not in (0,1) for row in masks for x in row):
        raise ValueError('Need a nonempty binary mask.')
    return sum(v*m for row,mask in zip(values,masks) for v,m in zip(row,mask))/denominator


def eos_mask(tokens,eos_id):
    result=[];seen=False
    for token in tokens:
        result.append(0 if seen else 1)
        seen=seen or token==eos_id
    return result


def report():
    logits=[0.,math.log(3)];old=math.log(.75)
    loss,gradient,details=policy_loss_and_gradient(logits,1,old,.5)
    numerical=finite_difference(logits,1,old,.5)
    if any(not math.isclose(a,b,abs_tol=1e-7) for a,b in zip(gradient,numerical)):
        raise ValueError('Analytic gradient and numerical gradient disagree.')
    changed=[z-.1*g for z,g in zip(logits,gradient)]
    positive_clip=policy_loss_and_gradient([0.,math.log(9)],1,math.log(.5),1.)
    negative_clip=policy_loss_and_gradient([0.,math.log(9)],1,math.log(.5),-1.)
    token_mean=masked_mean([[1.,1.,0.,0.,0.,0.],[-1.]*6],[[1,1,0,0,0,0],[1]*6])
    return {'scope':'Analytic differentiation + central finite difference + actual toy SGD of two scalar logits. No torch/model/tokenizer/Code-R1 optimizer execution.',
            'source_mapping':{'forward':'dp_actor.py::_forward_micro_batch','policy_loss':'core_algos.py::compute_policy_loss',
                              'autograd':'dp_actor.py::update_policy loss.backward; this lab uses an analytic derivative instead'},
            'one_token':{**details,'loss':loss,'analytic_gradient':gradient,'finite_difference_gradient':numerical,
                         'learning_rate':.1,'updated_logits':changed,'updated_selected_probability':softmax(changed)[1],
                         'frozen_values':['old_logprob','advantage'],'reference_policy':'omitted; isolate PG in this toy'},
            'temperature':{'T1_probability':softmax(logits,1.)[1],'T2_probability':softmax(logits,2.)[1]},
            'clip':{'positive_advantage':{'loss':positive_clip[0],'gradient':positive_clip[1],**positive_clip[2]},
                    'negative_advantage':{'loss':negative_clip[0],'gradient':negative_clip[1],**negative_clip[2]}},
            'mask_and_reduction':{'same_micro_batch_token_mean':token_mean,'per_response_then_mean':0.,
                                  'two_equal_weight_micro_batches':0.,'note':'The source normalizes within each micro batch, then accumulates. Not a global-mini-batch token average.',
                                  'eos_example':eos_mask([3,4,9,0,0],9),'no_eos_example':eos_mask([3,4,5,6,7],9),
                                  'pad_without_eos_example':eos_mask([3,4,0,0,0],9)},
            'manual_check':['Why freezing old prevents cancellation?', 'Why clip is not gradient-norm clipping?',
                            'Why reward need not be differentiable?', 'Why no-EOS mask cannot prove natural completion?']}


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',help='Optional exercise run; preserve report in a new attempt directory.')
    args=parser.parse_args();result=report()
    if args.run:
        import practice
        folder=practice.run_directory(args.run);destination=practice.attempt(folder)
        practice.save(destination,'mechanism_report.json',result)
        print('Saved: '+str(destination))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
