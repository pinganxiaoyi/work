void read(TString filename, TString outputFilename) {
    cout << "Starting the read function..." << endl;

    TFile *f1 = TFile::Open(filename);
    TTree *tree = (TTree*)f1->Get("HERD_CALO");
    TBranch *b_IsCMOS = tree->GetBranch("IsCMOS");
    TBranch *b_TriggerNumber = tree->GetBranch("TriggerNumber");

    double buffer[1029][3];
    b_IsCMOS->SetAddress(&buffer);

    int TriggerNumber;
    b_TriggerNumber->SetAddress(&TriggerNumber);
    std::vector<double> selectdata;
    selectdata.reserve(tree->GetEntries());

    TFile *f2 = new TFile(outputFilename, "RECREATE");

    TTree *t1 = new TTree("ISCMOS", "ISCMOS");
    double revalue, mean , sigma;
    TBranch *b_revalue = t1->Branch("revalue", &revalue);
    TBranch *b_mean = t1->Branch("mean", &mean);
    TBranch *b_sigma = t1->Branch("sigma", &sigma);

for(int iCount = 0; iCount < 1029; iCount++) {
     selectdata.clear(); 
        for(int iEntry = 0; iEntry < tree->GetEntries(); ++iEntry) {
            b_IsCMOS->GetEntry(iEntry);
            b_TriggerNumber->GetEntry(iEntry);
            
            if(TriggerNumber == 0) {
                double value = buffer[iCount][0];
                selectdata.push_back(value);
            }
        }
        
        double xmin = *std::min_element(selectdata.begin(), selectdata.end());
        double xmax = *std::max_element(selectdata.begin(), selectdata.end());
         
        TH1D *hist = new TH1D("hist",Form("Histogram of IsCMOS[%d][0]", iCount), 100, xmin, xmax);
        
        for(double val : selectdata) {
            hist->Fill(val);
        }
        
        TF1 *gaussian = new TF1("gaussian", "gaus", xmin, xmax);
        hist->Fit(gaussian, "0QR", "MULTITHREAD");
        Info("read",Form("fit success channel %d",iCount));
         mean = gaussian->GetParameter(1);
         sigma = gaussian->GetParameter(2);
         b_mean->Fill();
         b_sigma->Fill();
        for(double val : selectdata) {
            revalue = val - mean;  
            b_revalue->Fill();
            t1->Fill();
        }
        if (hist) {delete hist; hist = nullptr;} 
        if(gaussian) delete gaussian;
         
    }
    
    t1->Write();
      f2->Close();
        delete f2;
    f1->Close();
        delete f1;
    }
    
   









