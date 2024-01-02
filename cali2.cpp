#include <vector>
#include <string>
using namespace std;
void selectdata(){
 cout << "Starting the read function..." << endl;

    TFile *f1 = TFile::Open(ISCMOS_LR_RUN00093_BEAM_20230901_232333_00001.raw_512+2256x1024x11_HGain.raw.root);
    TTree *tree = (TTree*)f1->Get("HERD_CALO");
    TBranch *b_IsCMOS = tree->GetBranch("IsCMOS");
    TBranch *b_TriggerNumber = tree->GetBranch("TriggerNumber");
    TFile *f2 = TFile::Open(file1_v2.root);
    TTree *tree2 = (TTree*)f2->Get("ISCMOS");
    TBranch *b_mean = tree2->GetBranch("mean");
    TBranch *b_sigma = tree2->GetBranch("sigma");
    double buffer[1029][3],mean,sigma,sum;
    b_IsCMOS->SetAddress(&buffer);
    b_mean->SetAddress(&mean);
    b_sigma->SetAddress(&sigma);
    TBranch *b_sum = tree2->Branch("sum",&sum);
    int TriggerNumber;
    b_TriggerNumber->SetAddress(&TriggerNumber);
    std::vector<double> select_data;
    select_data.reserve(tree->GetEntries());
for(int iCount = 0; iCount < 1029; iCount++) {
     select_data.clear(); 
        for(int iEntry = 0; iEntry < tree->GetEntries(); ++iEntry) {
            b_IsCMOS->GetEntry(iEntry);
            b_TriggerNumber->GetEntry(iEntry);
            b_mean->GetEntry(iEntry);
            b_sigma->GetEntry(iEntry);
            if(TriggerNumber != 0) {
                double value = buffer[iCount][0];
                value = value - mean;
                select_data.push_back(value);
            }
        }
        
        double xmin = *std::min_element(select_data.begin(), select_data.end());
        double xmax = *std::max_element(select_data.begin(), select_data.end());
         
        TH1D *hist = new TH1D("hist",Form("Histogram of IsCMOS[%d][0]", iCount), 100, xmin, xmax);
        
        for(double val : select_data) {
            hist->Fill(val);
            sum += val ;
            b_sum->Fill();
        }
        delete hist , hist2;
}
}

        
    