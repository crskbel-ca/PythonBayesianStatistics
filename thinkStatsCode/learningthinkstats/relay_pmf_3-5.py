import relay
import Pmf
import myplot


def main():
    results = relay.ReadResults()
    speeds = relay.GetSpeeds(results)

    # plot the distribution of actual speeds
    pmf = Pmf.MakePmfFromList(speeds, 'speeds')

    myplot.Pmf(pmf)
    myplot.show(root='relay_pmf_3-5',
                title='PMF of running speed',
                xlabel='speed (mph)',
                ylabel='probability')


if __name__ == '__main__':
    main()