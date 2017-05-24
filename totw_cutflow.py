#!/usr/bin/env python

# @file:    totw_cutflow.py
# @purpose: easily make professional looking cutflows for different regions
# @author:  Joakim Olsson <joakim.olsson@cern.ch>
# @date:    April 2017
# (based on Giordon Stark's totw.py <gstark@cern.ch>)

# __future__ imports must occur at beginning of file
#   print(string, f=fd)
from __future__ import print_function
# used to redirect ROOT output
#   see http://stackoverflow.com/questions/21541238/get-ipython-doesnt-work-in-a-startup-script-for-ipython-ipython-notebook
import tempfile

import os, sys
# grab the stdout and have python write to this instead
# ROOT will write to the original stdout
STDOUT = os.fdopen(os.dup(sys.stdout.fileno()), 'w')

# for logging, set it up
import logging
root_logger = logging.getLogger()
root_logger.addHandler(logging.StreamHandler(STDOUT))
logger = logging.getLogger("totw")

# import all libraries
import argparse
import subprocess
import glob
import yaml
import json

'''
  with tempfile.NamedTemporaryFile() as tmpFile:
    if not args.debug:
      ROOT.gSystem.RedirectOutput(tmpFile.name, "w")

    # execute code here

    if not args.debug:
      ROOT.gROOT.ProcessLine("gSystem->RedirectOutput(0);")
'''

# Set up ROOT
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

#root_numpy
import numpy as np
import root_numpy as rnp
import rootpy as rpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from rootpy.io import root_open
from rootpy.plotting.style import set_style, get_style
from rootpy.plotting import Canvas, Legend, HistStack, Hist, Pad
from rootpy.plotting.hist import _Hist
from palettable import colorbrewer, tableau
from itertools import cycle, chain
import copy
import re

import plotHelpers as ph

def format_arg_value(arg_val):
  """ Return a string representing a (name, value) pair.

  >>> format_arg_value(('x', (1, 2, 3)))
  'x=(1, 2, 3)'
  """
  arg, val = arg_val
  return "%s=%r" % (arg, val)

# http://wordaligned.org/articles/echo
def echo(*echoargs, **echokwargs):
  logger.debug(echoargs)
  logger.debug(echokwargs)
  def echo_wrap(fn):
    """ Echo calls to a function.

    Returns a decorated version of the input function which "echoes" calls
    made to it by writing out the function's name and the arguments it was
    called with.
    """

    # Unpack function's arg count, arg names, arg defaults
    code = fn.func_code
    argcount = code.co_argcount
    argnames = code.co_varnames[:argcount]
    fn_defaults = fn.func_defaults or list()
    argdefs = dict(zip(argnames[-len(fn_defaults):], fn_defaults))

    def wrapped(*v, **k):
      # Collect function arguments by chaining together positional,
      # defaulted, extra positional and keyword arguments.
      positional = map(format_arg_value, zip(argnames, v))
      defaulted = [format_arg_value((a, argdefs[a]))
                   for a in argnames[len(v):] if a not in k]
      nameless = map(repr, v[argcount:])
      keyword = map(format_arg_value, k.items())
      args = positional + defaulted + nameless + keyword
      write("%s(%s)\n" % (fn.__name__, ", ".join(args)))
      return fn(*v, **k)
    return wrapped

  write = echokwargs.get('write', sys.stdout.write)
  if len(echoargs) == 1 and callable(echoargs[0]):
    return echo_wrap(echoargs[0])
  return echo_wrap

#@echo(write=logger.debug)
def do_totw(args):
  pass

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

did_regex = re.compile('(\d{6,8})')
def get_did(hist):
  if not isinstance(hist, _Hist):
    import pdb; pdb.set_trace()
    raise TypeError("Must pass in a rootpy Hist object")
  m = did_regex.search(hist.get_directory().GetFile().name)
  if m is None:
    raise ValueError("%s is not a valid filename" % hist.get_directory().GetFile().name)
  return m.groups()[0]

def add_files(groups):
  hall = ph.HChain(args.topLevel)
  for group in configs[groups]:
    hc = ph.HGroup(group['name'])
    logger.log(25, "Group: {0:s}".format(group['name']))
    for f in group['files']:
      logger.log(25, "\tPattern: {0:s}".format(f))
      for fname in glob.glob(f):
        logger.log(25, "\t\tAdding {0:s}".format(fname))
        rootFile = root_open(fname)
        hc.append(rootFile)
    if len(hc) == 0:
      raise ValueError("{0:s} has no files loaded.".format(group['name']))
    logger.log(25, "\tAdding {0:s}".format(hc))
    hall.append(hc)
  return hall

def scale_hists(h, weights, groups):
  # scale the histograms before doing anything else
  for hgroup in h:
    print (groups.get(hgroup.group))
    if groups.get(hgroup.group).get('do not scale me', False):
      logger.info("Skipping %s for scaling" % hgroup.group)
      continue
    for hist in hgroup:
      # scale the histograms, look up weights by did
      did = get_did(hist)
      weight = weights.get(did)
      if weight is None:
        raise KeyError("Could not find the weights for did=%s" % did)
      scaleFactor = 1.0
      if weight.get('num events') == 0:
        raise ValueError("did=%s has num events == 0" % did)
      scaleFactor /= weight.get('num events')
      scaleFactor *= weight.get('cross section')
      scaleFactor *= weight.get('filter efficiency')
      scaleFactor *= weight.get('k-factor')
      scaleFactor *= args.global_luminosity*1000
      scaleFactor *= groups.get(hgroup.group).get('scale factor', 1.0)
      hist.scale(scaleFactor)
      logger.info("Scale factor for %s: %0.6f" % (did, scaleFactor))

def save_cutflow(sig_hists, bkg_hists, title='cutflow', outfile=''):

  thickline = '*' * (160)
  line = '-' * (160)

  try:
      ensure_dir(outfile)
      f = open(outfile, 'w')

      print(thickline, file=f)
      print("Cutflow: {}".format(title), file=f)
      print(thickline, file=f)

      print("Background", file=f)
      print(line, file=f)
      print("{} | Num. Events             | W. Num. Events".format(" "*25), file=f)
      print(line, file=f)
      tot_bkg = 0.
      tot_bkg_err = 0.0
      w_tot_bkg = 0.
      w_tot_bkg_err = 0.0
      for h in bkg_hists:
          name = h.GetTitle()
          evt = h.GetEntries()
          err = np.sqrt(evt)
          wevt = h.GetBinContent(1)
          werr = h.GetBinError(1)
          tot_bkg += h.GetEntries()
          tot_bkg_err += err**2
          w_tot_bkg += h.GetBinContent(1)
          w_tot_bkg_err += werr**2
          print("{0: <25} | {1:10.2f} +/- {2:8.2f} | {3:10.2f} +/- {4:8.2f}".format(
              name, evt, err, wevt, werr), file=f)
      print(line, file=f)
      tot_bkg_err = np.sqrt(tot_bkg_err)
      w_tot_bkg_err = np.sqrt(w_tot_bkg_err)
      print("{0: <25} | {1:10.2f} +/- {2:8.2f} | {3:10.2f} +/- {4:8.2f}".format(
          "Tot. Background", tot_bkg, tot_bkg_err, w_tot_bkg, w_tot_bkg_err), file=f)
      print(thickline, file=f)
      print("Signal", file=f)
      print(line, file=f)
      # print("{} | Num. Events             | W. Num. Events          | S/sqrt(B_tot)   | S/sqrt(B_tot+1.5+(0.3*B_tot)^2) | BinomialExpZ(S,B_tot,0.30)".format(" "*25))
      print("{} | Num. Events             | W. Num. Events          | S/sqrt(B_tot)   | BinomialExpZ(S,B_tot,0.30)".format(" "*25), file=f)
      print(line, file=f)
      for h in sig_hists:
          name = h.GetTitle()
          evt = h.GetEntries()
          err = np.sqrt(evt)
          wevt = h.GetBinContent(1)
          werr = h.GetBinError(1)
          signif_1 = wevt/np.sqrt(w_tot_bkg)
          # signif_2 = wevt/np.sqrt(w_tot_bkg+1.5+(0.3*w_tot_bkg)**2)
          signif_2 = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(wevt, w_tot_bkg, 0.1)
          signif_3 = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(wevt, w_tot_bkg, 0.3)

          # testP = ROOT.RooStats.NumberCountingUtils.BinomialExpP(50, 100, 0.3)
          # testZ = ROOT.RooStats.NumberCountingUtils.BinomialExpZ(50, 100, 0.3)
          # print("testP", testP)
          # print("testZ", testZ)

          # print("{0: <25} | {1:10.2f} +/- {2:8.2f} | {3:10.2f} +/- {4:8.2f} | {5:15.2f} | {6:31.2f} | {7:10.2f}".format(
          #     name, evt, err, wevt, werr, signif_1, signif_2, signif_3))
          print("{0: <25} | {1:10.2f} +/- {2:8.2f} | {3:10.2f} +/- {4:8.2f} | {5:15.3f} | {6:10.3f}".format(
              name, evt, err, wevt, werr, signif_1, signif_3), file=f)
      print(thickline, file=f)
  except IOError:
      print ('Could not open file', outfile)


if __name__ == "__main__":
  class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    pass

  __version__ = subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(os.path.realpath(__file__))).strip()
  __short_hash__ = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.realpath(__file__))).strip()

  parser = argparse.ArgumentParser(description='Author: Giordon Stark. v.{0}'.format(__version__),
                                   formatter_class=lambda prog: CustomFormatter(prog, max_help_position=30))

  # general arguments for all
  parser.add_argument('-v','--verbose', dest='verbose', action='count', default=0, help='Enable verbose output of various levels. Use --debug to enable output for debugging.')
  parser.add_argument('--debug', dest='debug', action='store_true', help='Enable ROOT output and full-on debugging. Use this if you need to debug the application.')
  parser.add_argument('-b', '--batch', dest='batch_mode', action='store_true', help='Enable batch mode for ROOT.')

  parser.add_argument('--config', required=True, type=str, dest='config_file', metavar='<file.yml>', help='YAML file specifying input files and asssociated names')
  parser.add_argument('--weights', required=True, type=str, dest='weights_file', metavar='<file.json>', help='json file specifying the weights by dataset id')
  parser.add_argument('--lumi', required=False, type=int, dest='global_luminosity', metavar='<ifb>', help='luminosity to use for scaling')
  parser.add_argument("--do",                  help="Do systemtic variations [0/1]", type=int, default=0)

  parser.add_argument('-i', '--input', dest='topLevel', type=str, help='Top level directory containing plots.', default='all')

  # parse the arguments, throw errors if missing any
  args = parser.parse_args()

  try:
    # start execution of actual program
    import timing

    # set verbosity for python printing
    if args.verbose < 5:
      logger.setLevel(25 - args.verbose*5)
    else:
      logger.setLevel(logging.NOTSET + 1)

    with tempfile.NamedTemporaryFile() as tmpFile:
      if not args.debug:
        ROOT.gSystem.RedirectOutput(tmpFile.name, "w")

      # if flag is shown, set batch_mode to true, else false
      ROOT.gROOT.SetBatch(args.batch_mode)

      configs = yaml.load(file(args.config_file))
      # get the backgrounds and signals configuration
      backgrounds = dict([(group['name'], group) for group in configs['backgrounds']])
      signals = dict([(group['name'], group) for group in configs['signals']])
      # get the cutflows configuration
      cutflows = configs.get('cutflows')
      # paths to cutflow histograms
      cutflows_paths = cutflows.get('paths')

      # get the weights configurations for scaling
      weights = json.load(file(args.weights_file))

      # load all the signal and background files
      bkgall = add_files('backgrounds')
      sigall = add_files('signals')

      for path in cutflows_paths.keys():

        hbkg = reduce(lambda x,y: getattr(x, y, None), (item for item in path.split('/') if item != args.topLevel), bkgall)
        hsig = reduce(lambda x,y: getattr(x, y, None), (item for item in path.split('/') if item != args.topLevel), sigall)

        scale_hists(hbkg, weights, backgrounds)
        scale_hists(hsig, weights, signals)

        bkg_hists = map(lambda hgroup: hgroup.flatten, hbkg)
        sig_hists = map(lambda hgroup: hgroup.flatten, hsig)

        print(bkg_hists)
        exit()

        cutflow_label = cutflows_paths.get(hsig.path, {})['label']
        outfile = cutflows_paths.get(hsig.path, {})['outfile']
        save_cutflow(sig_hists, bkg_hists, cutflow_label, outfile)

      if not args.debug:
        ROOT.gROOT.ProcessLine("gSystem->RedirectOutput(0);")

  except Exception, e:
    # stop redirecting if we crash as well
    if not args.debug:
      ROOT.gROOT.ProcessLine("gSystem->RedirectOutput(0);")

    logger.exception("{0}\nAn exception was caught!".format("-"*20))

  # here we close it all
  map(lambda hgroup: map(lambda hist: hist.get_file().close(), hgroup), bkgall)
  map(lambda hgroup: map(lambda hist: hist.get_file().close(), hgroup), sigall)
