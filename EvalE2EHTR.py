#!/usr/bin/env python3

# Standard packages
import sys, os, argparse
import glob, math, re
import timeit
from itertools import groupby
import numpy as np
#from numpy import zeros, inf, array, argmin
from scipy.optimize import linear_sum_assignment
import unicodedata as ud

#from textdistance import levenshtein
# Modified fastwer package 
import fastwer

# To always have unbuffered output with the "print" 
import functools
print = functools.partial(print, flush=True)


DUMSYMB = "<DUMMY>"


def nrSpearmanDstHUN(idx1, idx2, S1, S2, N, dumFct=1):
    assert len(idx1) == len(idx2) and N > 0
    dst = sum(abs(i-j) if (S1[i]!=DUMSYMB and S2[j]!=DUMSYMB) else 0 \
                if (S1[i]==DUMSYMB and S2[j]==DUMSYMB) else dumFct \
                for i, j in zip(idx1, idx2))

    #return dst / ((N*N-(N%2))/2)
    return dst / ((N*N)//2 if N > 1 else 1)


def nrSpearmanDstHUN_woDmy(idx1, idx2, S1, S2, N, dumFct=1):
    assert len(idx1) == len(idx2) and N > 0
    lIdxs = []
    for i, j in zip(idx1, idx2):
        if (S1[i] == DUMSYMB and S2[j]==DUMSYMB): continue
        elif S1[i] == DUMSYMB: lIdxs.append([i,j,0,1])
        elif S2[j] == DUMSYMB: lIdxs.append([i,j,1,0])
        else: lIdxs.append([i,j,0,0])
    # Re-indexing reference words
    lIdxs.sort(key=lambda x:x[0]) 
    n=0;
    for x in lIdxs:
        if x[2] == 1:
            n += 1
            continue
        x[0] = x[0] - n
    # Re-indexing hypothesis words
    lIdxs.sort(key=lambda x:x[1]) 
    n=0;
    for x in lIdxs:
        if x[3] == 1:
            n += 1
            continue
        x[1] = x[1] - n

    dst = sum(dumFct if x[2]==1 or x[3]==1 else abs(x[0]-x[1]) for x in lIdxs)
    #for x in lIdxs: print(x)
    #print(dst,N)
    return dst / ((N*N)//2 if N > 1 else 1)


def nrSpearmanDstWER(aligLst, dumFct=1):
    N = len(aligLst)
    assert N > 0
    dst = sum(dumFct if x[0]==-1 or x[1]==-1 else abs(x[0]-x[1]) for x in aligLst)
    return dst / ((N*N)//2 if N > 1 else 1)


def nrSpearmanDstWER_woDmy(aligLst, dumFct=1):
    N = len(aligLst)
    assert N > 0
    lIdxs = [[i,j,0,1] if i == -1 else [i,j,1,0] if j == -1 else [i,j,0,0] for i, j, _ in aligLst]
    # Re-indexing reference words
    n=0;
    for x in lIdxs:
        if x[2] == 1:
            n += 1
            continue
        x[0] = x[0] - n
    # Re-indexing hypothesis words
    n=0;
    for x in lIdxs:
        if x[3] == 1:
            n += 1
            continue
        x[1] = x[1] - n
    
    dst = sum(dumFct if x[2]==1 or x[3]==1 else abs(x[0]-x[1]) for x in lIdxs)
    #for x in lIdxs: print(x,N)
    #print(dst,N)
    return dst / ((N*N)//2 if N > 1 else 1)


def nrDifferentialDstHUN(idx1, idx2, S1, S2, N):
    assert len(idx1) == len(idx2) and N > 0
    gen = zip(idx1, idx2)
    #ip, jp = next(gen)
    ip, jp = 0, 0
    dst = 0
    for i, j in gen:
        if (S1[i]==DUMSYMB and S2[j]==DUMSYMB): continue
        elif (S1[i]==DUMSYMB or S2[j]==DUMSYMB): 
            dst += 1
            if S1[i]==DUMSYMB: jp = j
            else: ip = i
            continue
        else: 
            dst += abs((i-ip) - (j-jp))
            ip, jp = i, j
    return dst / (((N-1)*N/2 + (N%2 - 1)) if N > 2 else 1)


def reorder_align(idx1, idx2, cad2, cm):
    aux = [(k,j,cad2[j],cm[i,j]) for k, (i, j) in enumerate(zip(idx1, idx2))]
    aux.sort(key=lambda x:x[2:])
    #print(aux); sys.exit()
    for key, group in groupby(aux, lambda x: x[2:]):
        group = [e[0:2] for e in group]
        if len(group) == 1: continue
        idxs = list(zip(*group))[0]
        p_cur = sorted(group, key=lambda x:x[1])
        #print(key, idxs, p_cur)
        for n, i in enumerate(idxs): idx2[i] = p_cur[n][1]
    return idx2


def reinsert_words(idxIns, idxList):
    # Ignore indices of words to be reinserted in the list
    l2 = [e for e in idxList.tolist() if not(e in idxIns)]  
    for i in idxIns:
        if i > 0: 
            min, pos = sys.maxsize, len(l2)
            for n, j in enumerate(l2):
                if j<i and i-j<min: min, pos = i-j, n + 1
        else: pos = 0
        l2.insert(pos,i)
    #print(l2)
    return l2

    
def evalHungarian(X, Y, ctLn=0, addDum=False, fullDum=False, fcInsDel=1.0, fctRegul=0.0, sortAlign=False, reIns=False, verbTimes=0):
    
    lx, ly, lenW = len(X), len(Y), len(X)
    if verbTimes==1: print(f'\n#RWsRef: {lx}        #RWsHyp: {ly}\n')
    mlen, dlen = max(lx, ly), abs(lx - ly)
    
    ti = timeit.default_timer()
    if ctLn == 0:
        #cost = [int(x!=y) for x in X for y in Y]
        #cost = [levenshtein(x,y) for x in X for y in Y]
        cost = [fastwer.compute(x, y, char_level=True)[0] + abs(i-j)/mlen*fctRegul \
                for i, x in enumerate(X) for j, y in enumerate(Y)]
    else:
        cost = [fastwer.compute( \
         (' '.join(X[i:]+X[:i])[-ctLn:])+x+(' '.join(X[i+1:]+X[:i+1])[:ctLn]), \
         (' '.join(Y[j:]+Y[:j])[-ctLn:])+y+(' '.join(Y[j+1:]+Y[:j+1])[:ctLn]), \
         char_level=True)[0] + abs(i-j)/mlen*fctRegul \
         for i, x in enumerate(X) for j, y in enumerate(Y)]
    cost_matrix = np.array(cost, dtype=float).reshape((lx, ly))
    #print(cost_matrix.shape)

    dfLen = 0
    if addDum:
        dfLen = lx - ly
        if dfLen < 0:
            dumCost = np.array([(len(y)+ctLn*2)*fcInsDel + 1/mlen*fctRegul \
                                for x in range(-dfLen) \
                                for y in Y], dtype=float).reshape((-dfLen, ly))
            X += [DUMSYMB] * -dfLen
            cost_matrix = np.concatenate((cost_matrix, dumCost), axis=0)
        elif dfLen > 0:
            dumCost = np.array([(len(x)+ctLn*2)*fcInsDel + 1/mlen*fctRegul \
                                for x in X \
                                for y in range(dfLen)], dtype=float).reshape((lx, dfLen))
            Y += [DUMSYMB] * dfLen
            cost_matrix = np.concatenate((cost_matrix, dumCost), axis=1)
        lx, ly = len(X), len(Y)
        assert lx == ly
    #print(cost_matrix.shape)

    if addDum and fullDum:
        dumCostX = np.array([(len(y)+ctLn*2)*fcInsDel + 1/mlen*fctRegul \
                             if y!=DUMSYMB else 0 \
                             for x in range(lx) for y in Y], \
                            dtype=float).reshape((lx, ly))
        cost_matrix = np.concatenate((cost_matrix, dumCostX), axis=0)
        #print(cost_matrix.shape)
        
        dumCostY = np.array([(len(x)+ctLn*2)*fcInsDel + 1/mlen*fctRegul \
                             if x!=DUMSYMB else 0 \
                             for x in X for y in range(ly)], \
                            dtype=float).reshape((lx, ly))
        dumCostD = np.zeros((lx, ly), dtype=int)
        dumCostY = np.concatenate((dumCostY, dumCostD), axis=0)
        
        cost_matrix = np.concatenate((cost_matrix, dumCostY), axis=1)
        X += [DUMSYMB] * lx; Y += [DUMSYMB] * ly
        lx, ly = len(X), len(Y)
        assert lx == ly
    #print(cost_matrix.shape); sys.exit()
    if verbTimes==2: print("TIME CST_mat: Cost matrix building computation: %d (wrds) %f (sec)" % (lenW,timeit.default_timer()-ti), file=sys.stderr)

    # Apply Hungarian's algorithms
    # See: https://en.wikipedia.org/wiki/Hungarian_algorithm
    ti = timeit.default_timer()
    r1, r2 = linear_sum_assignment(cost_matrix)
    if verbTimes==2: print("TIME HUN_alg: Hungarian algorithm computation: %d (wrds) %f (sec)" % (lenW,timeit.default_timer()-ti), file=sys.stderr)

    if sortAlign:
        # This does not affect the Hungarian's cost
        r2 = reorder_align(r1, r2, Y, cost_matrix)

    sumT = hErr = dmyErr = 0
    for i, j in zip(r1, r2):
        if X[i] == Y[j] == DUMSYMB: continue
        #print(f'({i:03d}) {X[i]:15} ---> ({j:03d}) {Y[j]:15} ---> {cost_matrix[i,j]}')
        if verbTimes==1: print(f'{i:3d} {X[i]:15} ---> {j:3d} {Y[j]:15} ---> {cost_matrix[i,j]:4.2f}')
        hErr += X[i] != Y[j]
        dmyErr += 1 if (X[i] == DUMSYMB or Y[j] == DUMSYMB) else 0
        sumT += cost_matrix[i,j]
    #print('-'*55+'\n'+f'Hungarian\'s Cost: {cost_matrix[r1, r2].sum():>35}({cost_matrix[r1, r2][>0].sum():>35})')
    # Convert Ins+Del into Sust for the DUMMY entries not avoidable
    hErr -= int((dmyErr - dlen) // 2)
    if verbTimes==1: print('-'*55+'\n'+f'Hungarian\'s Cost: {sumT:36.2f}    #wrdErr: {hErr}')
    
    numInsWrds = numInsChrs = numInsWrds_wDumm = 0
    if lx > ly:
        assert ly == len(r1) == len(r2)
        idxs = list(set([e for e in range(lx)]) - set(r1))
        ex_err = len(' '.join([X[i] for i in idxs])) + 1
        if verbTimes==1: print('\nUnaligned words of Ref:',idxs,'-->',str([X[i] for i in idxs]), '--> #Del-Chars:', ex_err)
    elif ly > lx:
        assert lx == len(r1) == len(r2)
        idxs = list(set([e for e in range(ly)]) - set(r2))
        ex_err = len(' '.join([Y[i] for i in idxs])) + 1
        numInsWrds, numInsChrs = len(idxs), ex_err
        if verbTimes==1: print('\nUnaligned words of Hyp:',idxs,'-->',str([Y[i] for i in idxs]), '--> #Ins-Chars: ', ex_err)
    else:
        # Looking for inserted words when dummy words are used
        idxs = [j for i, j in zip(r1, r2) if X[i] == DUMSYMB and Y[j] != DUMSYMB]
        numInsWrds_wDumm = len(idxs)
        if numInsWrds_wDumm and verbTimes==1:
            print('\nUnaligned words of Hyp using DUMMY words:',idxs,'-->',str([Y[i] for i in idxs]))
        ex_err = abs(dfLen) # Take into account the number of
                            # deleted/inserted white-spaces of dummy
                            # words

    aligDsts = (nrSpearmanDstHUN_woDmy(r1, r2, X, Y, mlen), \
                nrDifferentialDstHUN(r1, r2, X, Y, mlen))
    
    if (numInsWrds > 0 or numInsWrds_wDumm > 0) and reIns:
        l2 = reinsert_words(idxs, r2)
        strHyp = ' '.join([Y[i] for i in l2 if Y[i] != DUMSYMB])
        numInsWrds, numInsChrs = 0, 0
    else:
        strHyp = ' '.join([Y[i] for i in r2 if Y[i] != DUMSYMB])

    return hErr, cost_matrix[r1, r2].sum() + ex_err, strHyp, numInsWrds, numInsChrs, aligDsts


def intConfComp(nS, cte, *karg):
    res = list()
    for arg in karg:
        try:
            k = cte * math.sqrt( (arg*(1-arg)) / nS)
        except ValueError:
            k = 0
        res.append(k)
    return tuple(res)


def compute_BOW(x, y):
    err, acc, _ = fastwer.bagOfWords(y, x, char_level=False)
    return err, acc


def computeSIDerr(alignLvd):
    s = i = d = c = 0
    if len(alignLvd) == 0: return (0,0,0)
    for l in alignLvd:
        if l[2] == 1:
            if l[0] == -1: d+=1
            elif l[1] == -1: i+=1
            else: s+=1
        else: c+=1
    return (s,i,d,c)


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"ERROR: Directory {path} is not a valid !")

    
def main():
    parser = argparse.ArgumentParser(description='This program evaluates performance of an end-to-end HTR output.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbosity', type=int, choices=[0, 1, 2], help='If it is set to 1, print results per page; if it is set to 2, compute also running time for each processing step.', default=0)
    parser.add_argument('-H', '--hung', action='store_true', help='Compute also Hungarian WER and CER (WER_hun and CER_hlv)', default=False)
    parser.add_argument('-r', '--reIns', action='store_true', help='Reinsert non-aligned words of the hypotheses into the aligned ones', default=False)
    parser.add_argument('-s', '--sortAlign', action='store_true', help='Sort hypothesis words with the same alignment cost by their positional index', default=False)
    parser.add_argument('--refExt', type=str, help='File extension of reference transcripts', default='ref')
    parser.add_argument('--hypExt', type=str, help='File extension of predicted transcripts', default='hyp')
    parser.add_argument('-c', '--ctxtLeng', type=int, help='Number of characters taken from the left and right context (words) of the current word, which are attached to it to compute its Lv-Dst', default=0)
    parser.add_argument('-d', '--addDummies', action='store_false', help='Not add dummy words when number of words between reference and hypothesis differs (this enables insertions or deletions)', default=True)
    parser.add_argument('-x', '--insDelFactor', type=float, help='Factor to weigh insertion/deletions costs whem DUMMY words are used.', default=0.5)
    parser.add_argument('-g', '--regFactor', type=float, help='Regularization factor to weight relative word positions in the alignment computation of Hungarian algorithm.', default=1.0)
    parser.add_argument('-f', '--fullDummies', action='store_false', help='Not add dummy words to the references and hypotheses (this enables insertions and deletions simultaneously)', default=True)
    parser.add_argument('dataDir', type=dir_path, help='Directory of files containing reference transcripts (.ref) and corresponding predicted transcripts (.hyp).')

    args = parser.parse_args()
    if args.ctxtLeng < 0:
        parser.error('Number of contextual chars must a positive integer !')
    if args.insDelFactor < 0:
        parser.error('Factor value to weigh insertion/deletions costs must a positive float !')
    if args.regFactor < 0:
        parser.error('Factor value for regularization cost must a positive float !')
    #if args.reIns and args.addDummies:
    #    parser.error(f'"reIns" and "addDummies" are incompatible together !')
    if args.fullDummies and not args.addDummies:
        parser.error(f'"addDummies" must also be enable to use "fullDummies" !')
    if args.insDelFactor!=1.0 and not (args.addDummies and args.fullDummies):
        parser.error(f'"addDummies" and "fullDummies" must also be enable to use "insDelFactor" !')
    print('Running parameters: {}'.format(args), file=sys.stdout)
    verbT = args.verbosity

    glob_lvd_wl_org = glob_lvd_cl_org = 0
    glob_lvd_wl_alg = glob_lvd_cl_alg = 0
    glob_SIDerr, glob_SIDerrBoW = [0,0,0,0], [0,0,0,0]
    glob_lvd_wl_hun = glob_lvd_cl_hun = 0
    glob_ref_wl = glob_ref_cl = 0
    glob_bow_alg_df = glob_bow_alg_df_sust = 0
    nSamples = 0
    glob_dst0_alg = glob_dst1_alg = glob_dst2_alg = 0.0
    #print("\nProcessing Directory: \"{}\"\n".format(args.dataDir), file=sys.stderr)
    for rfile in glob.glob(os.path.join(args.dataDir, '*.'+args.refExt)):
        hfile = os.path.splitext(rfile)[0]+'.'+args.hypExt
        if not os.path.exists(hfile):
            print(f'WARNING: File \"{hfile}\" does not exist ! Skipping ...', file=sys.stderr)
            continue
        if verbT==1: print('\n'+'*'*80)
        if verbT==1: print("Processing files:\n\"{}\"\n\"{}\"".format(rfile, hfile), file=sys.stdout)
        if verbT==2: print("\nPROCESSING: \"{}\", \"{}\"".format(rfile, hfile), file=sys.stderr)

        nSamples += 1
        with open(rfile, 'r') as ref, open(hfile, 'r') as hyp:
            x = ud.normalize('NFC', ref.read().strip())
            x = re.sub(r'\s+',' ',x)
            y = ud.normalize('NFC', hyp.read().strip())
            y = re.sub(r'\s+',' ',y)
            #print(y); sys.exit(1)
            if verbT==1: print(f'REF: {x}'); print(f'HYP: {y}')
        x_s, y_s = x.split(), y.split()

        if args.hung:
            hWER, scr, hyp, niw, nic, dsts = evalHungarian(x_s.copy(), y_s.copy(), \
                                                     ctLn=args.ctxtLeng, \
                                                     addDum=args.addDummies, \
                                                     fullDum=args.fullDummies, \
                                                     fcInsDel=args.insDelFactor, \
                                                     fctRegul=args.regFactor, \
                                                     sortAlign=args.sortAlign, \
                                                     reIns=args.reIns,
                                                     verbTimes=verbT)
            if verbT==1: print(f'\nHYP-ALG: {hyp}')
            
        lenW, lenC = len(x_s), len(x)
        glob_ref_wl += lenW;  glob_ref_cl += lenC

        ti = timeit.default_timer()
        #lvd_cl_org = levenshtein(x, y)
        lvd_cl_org, lC = fastwer.compute(y, x, char_level=True)
        if verbT==2: print("TIME CER_lev: Levenshtein distance computation at char level: %d (chars) %f (sec)" % (lenC,timeit.default_timer()-ti), file=sys.stderr)
        ti = timeit.default_timer()
        #lvd_wl_org = levenshtein(x_s, y_s)
        #lvd_wl_org, lW = fastwer.compute(y, x, char_level=False)
        wl_algn_org, lvd_wl_org, lW = fastwer.alignment(y, x, char_level=False)
        SIDerr = computeSIDerr(wl_algn_org)
        glob_SIDerr[0]+=SIDerr[0]; glob_SIDerr[1]+=SIDerr[1]
        glob_SIDerr[2]+=SIDerr[2]; glob_SIDerr[3]+=SIDerr[3]
        if verbT==2: print("TIME WER_lev: Levenshtein distance computation at word level: %d (wrds) %f (sec)" % (lenW,timeit.default_timer()-ti), file=sys.stderr)
        # Sanity check
        assert (lW==lenW and lC==lenC), \
            'ERROR: Inconsistency with reference lengths !'
        glob_lvd_wl_org += lvd_wl_org; glob_lvd_cl_org += lvd_cl_org
        if verbT==1:
            print(f'\n   WER_lev: {lvd_wl_org:4d} {lenW:6d} {lvd_wl_org/lenW*100:6.2f}%    (S:{SIDerr[0]} I:{SIDerr[1]} D:{SIDerr[2]} C:{SIDerr[3]})')
            print(f'   CER_lev: {lvd_cl_org:4d} {lenC:6d} {lvd_cl_org/lenC*100:6.2f}%')

        ti = timeit.default_timer()
        bow_alg, acc_bow = compute_BOW(y, x)
        if verbT==2: print("TIME WER_bow: Bag-of-word WER computation: %d (wrds) %f (sec)" % (lenW,timeit.default_timer()-ti), file=sys.stderr)
        df = (len(x_s)-len(y_s)); dfa = abs(df)
        glob_bow_alg_df += (bow_alg - dfa) // 2 + dfa
        glob_bow_alg_df_sust += (bow_alg - dfa) // 2
        glob_SIDerrBoW[0]+=(bow_alg-dfa)//2;
        if df<=0:
            sid=f"S:{(bow_alg-dfa)//2} I:{dfa} D:0 C:{acc_bow}"
            glob_SIDerrBoW[1]+=dfa
        else:
            sid=f"S:{(bow_alg-dfa)//2} I:0 D:{dfa} C:{acc_bow}"
            glob_SIDerrBoW[2]+=dfa
        glob_SIDerrBoW[3]+=acc_bow
        if verbT==1:
            print(f'\n   WER_bow: {(bow_alg-dfa)//2 + dfa:4d} {lenW:6d} {((bow_alg-dfa)//2 + dfa) / lenW*100:6.2f}%    ({sid})')


        if args.hung:        
            ti = timeit.default_timer() 
            lvd_cl_alg = fastwer.compute(hyp, x, char_level=True)[0] + nic # Including inserted chars
            if verbT==2: print("TIME CER_hun: Levenshtein distance computation at char level: %d (chars) %f (sec)" % (lenC,timeit.default_timer()-ti), file=sys.stderr)
            ti = timeit.default_timer()
            lvd_wl_alg = fastwer.compute(hyp, x, char_level=False)[0] + niw # Including inserted words
            if verbT==2: print("TIME WER_hun: Levenshtein distance computation at word level: %d (wrds) %f (sec)" % (lenW,timeit.default_timer()-ti), file=sys.stderr)
            glob_lvd_wl_alg += lvd_wl_alg; glob_lvd_cl_alg += lvd_cl_alg
            glob_lvd_wl_hun += hWER
            if verbT==1:
                print(f'\n   WER_hun: {hWER:4d} {lenW:6d} {hWER/lenW*100:6.2f}%')
                print(f'   WER_hlv: {lvd_wl_alg:4d} {lenW:6d} {lvd_wl_alg/lenW*100:6.2f}%')
                print(f'   CER_hlv: {lvd_cl_alg:4d} {lenC:6d} {lvd_cl_alg/lenC*100:6.2f}%')

            dsts_lvDst = nrSpearmanDstWER_woDmy(wl_algn_org)
            glob_dst2_alg += dsts_lvDst*lenW
            glob_dst0_alg += dsts[0]*lenW; glob_dst1_alg += dsts[1]*lenW
            
            if verbT==1:
                print(f'\n   SPR_lev: {dsts_lvDst:.6f} (Normalized Spearman\'s footrule distance)')
                print(f'   SPR_hun: {dsts[0]:.6f} (Normalized Spearman\'s footrule distance)')
                print(f'   DIF_hun: {dsts[1]:.6f} (Normalized Differential distance)')

            if args.ctxtLeng == 0 and args.insDelFactor == 1.0:
                glob_lvd_cl_hun += int(scr)
                print(f'\n   CER_cst: {int(scr):4d} {len(x):6d} {scr/len(x)*100:6.2f}%')

            
    g_wer_org, g_cer_org, g_bow_alg, g_wer_hun, g_wer_alg, g_cer_alg, \
    g_cer_hun = \
        glob_lvd_wl_org/glob_ref_wl, glob_lvd_cl_org/glob_ref_cl, \
        glob_bow_alg_df/glob_ref_wl, glob_lvd_wl_hun/glob_ref_wl, \
        glob_lvd_wl_alg/glob_ref_wl, \
        glob_lvd_cl_alg/glob_ref_cl, glob_lvd_cl_hun/glob_ref_cl
    #print(g_wer_org, g_cer_org, g_wer_alg, g_cer_alg, g_cer_hun)
    
    ic_wer_org, ic_bow_alg, ic_wer_hun, ic_wer_alg = \
                                intConfComp(glob_ref_wl, 1.96, \
                                    g_wer_org, g_bow_alg, g_wer_hun, g_wer_alg)
    ic_cer_org, ic_cer_alg, ic_cer_hun = intConfComp(glob_ref_cl, 1.96, \
                                            g_cer_org, g_cer_alg, g_cer_hun)
    #print(ic_wer_org, ic_cer_org, ic_wer_alg, ic_cer_alg, ic_cer_hun); sys.exit()

    print(f'\n\nNumber of processed Samples: {nSamples}')
    print('='*50)
    print(f'G-WER_lev: {glob_lvd_wl_org:6d} {glob_ref_wl:6d} {g_wer_org*100:6.2f}% ±({ic_wer_org*100:.2f}%)    (S:{glob_SIDerr[0]} I:{glob_SIDerr[1]} D:{glob_SIDerr[2]} C:{glob_SIDerr[3]})')
    print(f'G-CER_lev: {glob_lvd_cl_org:6d} {glob_ref_cl:6d} {g_cer_org*100:6.2f}% ±({ic_cer_org*100:.2f}%)\n')

    print(f'G-WER_bow: {glob_bow_alg_df:6d} {glob_ref_wl:6d} {g_bow_alg*100:6.2f}% ±({ic_bow_alg*100:.2f}%)    (S:{glob_SIDerrBoW[0]} I:{glob_SIDerrBoW[1]} D:{glob_SIDerrBoW[2]} C:{glob_SIDerrBoW[3]})')

    if args.hung:
        print(f'\nG-WER_hun: {glob_lvd_wl_hun:6d} {glob_ref_wl:6d} {g_wer_hun*100:6.2f}% ±({ic_wer_hun*100:.2f}%)')
        print(f'G-WER_hlv: {glob_lvd_wl_alg:6d} {glob_ref_wl:6d} {g_wer_alg*100:6.2f}% ±({ic_wer_alg*100:.2f}%)')
        print(f'G-CER_hlv: {glob_lvd_cl_alg:6d} {glob_ref_cl:6d} {g_cer_alg*100:6.2f}% ±({ic_cer_alg*100:.2f}%)')

        print(f'\nG-SPR_lev: {glob_dst2_alg / glob_ref_wl:.6f} (Average normalized Spearman\'s footrule distance)')
        print(f'G-SPR_hun: {glob_dst0_alg / glob_ref_wl:.6f} (Average normalized Spearman\'s footrule distance)')
        print(f'G-DIF_hun: {glob_dst1_alg / glob_ref_wl:.6f} (Average normalized Differential distance)')

        if args.ctxtLeng == 0 and args.insDelFactor == 1.0:
            print(f'\nG-CER_cst: {glob_lvd_cl_hun:4d} {glob_ref_cl:6d} {g_cer_hun*100:6.2f}% ±({ic_cer_hun*100:.2f}%)')
    print('='*50)
    

    
if __name__ == "__main__":
    main()
    sys.exit(os.EX_OK)
