let Crawler = require("js-crawler");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;
const arr = [];
for (let i = 1; i <= 279; i++) {
  new Crawler()
    .configure({
      shouldCrawl: function (url) {
        return url === "https://snyk.io/vuln/page/" + i;
      },
    })
    .crawl("https://snyk.io/vuln/page/" + i, function onSuccess(page) {
      arr.push(page.content);
      if (arr.length === 279) {
        let out = [];
        arr.forEach((page) => {
          const firstSplit = page.split("<tbody>");
          const secondSplit = firstSplit[1].split("</tbody>");
          const thirdSplit = secondSplit[0].split("<tr >");
          out = [
            ...out,
            ...thirdSplit.map((el) => {
              // get sev label
              const sevLabelHalf = el.split(
                '<span class="severity-list__item-text">'
              );
              let sevLabel = "";
              sevLabelHalf.forEach((part, i) => {
                if (i === 1) {
                  sevLabel = part.split("</span>")[0];
                }
              });

              // get vuln name
              const vulnNameHalf = el.split("<strong >");
              let vulnName = "";
              vulnNameHalf.forEach((part, i) => {
                if (i === 1) {
                  vulnName = part.split("</strong>")[0];
                }
              });

              // get package name
              const pacNameHalf = el.split('<a  href="');
              let pacName = "";
              pacNameHalf.forEach((part, i) => {
                if (i === 2) {
                  part.split(">").forEach((p, t) => {
                    if (t === 1) {
                      pacName = p.split("</a")[0];
                    }
                  });
                }
              });

              // getPackage versions
              const pacVerHalf = el.split('<span  class="semver">');
              let pacVer = "";
              pacVerHalf.forEach((part, i) => {
                if (i === 1) {
                  pacVer = part.split("</span>")[0];
                }
              });

              // get type
              const typeHalf = el.split('<td  class="t--sm">');
              let type = "";
              typeHalf.forEach((part, i) => {
                if (i === 1) {
                  const typeSHalf = part.split("</td>");
                  typeSHalf.forEach((p, t) => {
                    if (t === 0) {
                      const typeArr = p.split(" ");
                      typeArr.forEach((chars) => {
                        if (chars.length > 1) {
                          type = chars.substring(0, chars.length - 1);
                        }
                      });
                    }
                  });
                }
              });

              // get publish date
              const dateHalf = el.split('<td  class="l-align-right t--sm">');
              let date = "";
              dateHalf.forEach((part, i) => {
                if (i === 1) {
                  const dateSHalf = part.split("</td>");
                  dateSHalf.forEach((p, t) => {
                    if (t === 0) {
                      const dateArr = p.split(" ");
                      // console.log(typeArr);
                      dateArr.forEach((chars) => {
                        if (chars.length > 1) {
                          date +=
                            date.length === 0
                              ? chars.substring(0, chars.length - 1)
                              : "," + chars.substring(0, chars.length - 1);
                        }
                      });
                    }
                  });
                }
              });

              return {
                label: sevLabel,
                vulnName: vulnName,
                packageName: pacName,
                packageVersion: pacVer,
                type: type,
                publishDate: date,
              };
            }),
          ];
        });

        console.log(out.length, arr.length);
        const csvWriter = createCsvWriter({
          path: "out.csv",
          header: [
            { id: "label", title: "Label" },
            { id: "vulnName", title: "Vulnerability Name" },
            { id: "packageName", title: "Package Name" },
            { id: "packageVersion", title: "Package Version" },
            { id: "type", title: "Type" },
            { id: "publishDate", title: "Publish Date" },
          ],
        });

        csvWriter
          .writeRecords(out)
          .then(() => console.log("The CSV file was written successfully"));
      }
    });
}
