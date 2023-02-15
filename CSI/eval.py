from common.eval import *

model.eval()

if P.mode == 'test_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, logger=None)

elif P.mode == 'test_marginalized_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

elif P.mode in ['ood', 'ood_pre']:
    if P.mode == 'ood':
        from evals import eval_ood_detection
    else:
        from evals.ood_pre import eval_ood_detection

    import os
    from evals import test_classifier
    with torch.no_grad():
        acc = 100 - test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

    with torch.no_grad():
        auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                        train_loader=train_loader, simclr_aug=simclr_aug)

    if P.one_class_idx is not None:
        mean_dict = dict()
        for ood_score in P.ood_score:
            mean = 0
            for ood in auroc_dict.keys():
                mean += auroc_dict[ood][ood_score]
            mean_dict[ood_score] = mean / len(auroc_dict.keys())
        auroc_dict['one_class_mean'] = mean_dict

    bests = []
    for ood in auroc_dict.keys():
        message = ''
        outstr = ''
        best_auroc = 0
        for ood_score, (auroc, tnr) in auroc_dict[ood].items():
            message += '[%s %s %.4f %.4f] ' % (ood, ood_score, auroc, tnr)
            outstr += ',%.5f,%.5f' % (auroc, tnr)
            if auroc > best_auroc:
                best_mode = ood_score
                best_auroc = auroc
                best_tnr = tnr
        message += '[%s %s %.4f %.4f] ' % (ood, 'best', best_auroc, best_tnr)
        if P.print_score:
            print(message)
        #bests.append((best_auroc,best_tnr))

        best_str = f'{P.seed},{acc:.4f},{best_mode},{best_auroc:.4f},{best_tnr:.4f}'
        outstr = best_str + outstr

        out_file = os.path.join("results",f'{P.dataset}_{ood}_{P.model}_{P.loss}.csv')
        with open(out_file, "a") as writer:
            writer.write(outstr+"\n")

    #bests = map('{:.4f} {:.4f}'.format, bests)
    #print('\t'.join(bests))

else:
    raise NotImplementedError()


